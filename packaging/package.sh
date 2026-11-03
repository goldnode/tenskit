#!/usr/bin/env bash
# Package a finished release build into target/dist: a .deb, an .rpm and a .tar.gz.
#
#   packaging/package.sh <rust-target> <amd64|arm64>
#
# Run from the repository root, after
# `cargo build --locked --release --target <rust-target>`. Needs nfpm on PATH.
set -euo pipefail

target=$1
arch=$2
tools=(tenspile tensinfo tensdiff tensconv tensquant)

version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)
[ -n "$version" ] || { echo "no version found in Cargo.toml" >&2; exit 1; }

release_dir=${CARGO_TARGET_DIR:-target}/$target/release
stage=target/package-stage
dist=target/dist
rm -rf "$stage" "$dist"
mkdir -p "$stage/bin" "$stage/man" "$dist"

for tool in "${tools[@]}"; do
    binary=$release_dir/$tool

    # A dynamic binary would break the promise that one package fits every distro.
    if readelf -d "$binary" | grep -q NEEDED || readelf -l "$binary" | grep -q INTERP; then
        echo "$binary is dynamically linked" >&2
        exit 1
    fi
    install -m 0755 "$binary" "$stage/bin/$tool"

    # build.rs writes these with @VERSION@ filled in; man/ still has the placeholder.
    gzip -9 -n -c "$release_dir/share/man/man1/$tool.1" > "$stage/man/$tool.1.gz"
done

export ARCH=$arch VERSION=$version
nfpm package --config packaging/nfpm.yaml --packager deb --target "$dist/"
nfpm package --config packaging/nfpm.yaml --packager rpm --target "$dist/"

# For distros that use neither format.
case $arch in
    amd64) machine=x86_64 ;;
    arm64) machine=aarch64 ;;
    *) echo "unknown arch $arch" >&2; exit 1 ;;
esac
name=tenskit-$version-$machine-linux
tree=$stage/$name
mkdir -p "$tree/bin" "$tree/share/man/man1"
cp "$stage"/bin/* "$tree/bin/"
cp "$stage"/man/* "$tree/share/man/man1/"
cp LICENSE README.md "$tree/"
tar -C "$stage" --owner=0 --group=0 -czf "$dist/$name.tar.gz" "$name"

ls -l "$dist"
