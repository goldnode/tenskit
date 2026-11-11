#!/bin/sh
# Install the package for this machine with the distro's own tool, then check it.
# Runs as root inside a distro container, from the repository root:
#
#   packaging/smoke-test.sh <directory holding the packages>
#
# POSIX sh, because not every distro image is guaranteed to have bash.
set -eu

dist=$1
tools="tenspile tensinfo tensdiff tensconv tensquant"
version=$(sed -n 's/^version = "\(.*\)"$/\1/p' Cargo.toml | head -n 1)

fail() {
    echo "FAILED: $*" >&2
    exit 1
}

case $(uname -m) in
    x86_64) deb_arch=amd64 rpm_arch=x86_64 ;;
    aarch64) deb_arch=arm64 rpm_arch=aarch64 ;;
    *) fail "unknown machine $(uname -m)" ;;
esac

# Paths start with ./ so that apt and dnf read a file instead of searching repos.
if command -v apt-get >/dev/null 2>&1; then
    pkg=./$dist/tenskit_${version}-1_${deb_arch}.deb
    [ -f "$pkg" ] || fail "$pkg is missing"
    listing=$(dpkg -c "$pkg")
    [ "$(dpkg-deb -f "$pkg" Architecture)" = "$deb_arch" ] || fail "wrong architecture in $pkg"
    license=usr/share/doc/tenskit/copyright
    DEBIAN_FRONTEND=noninteractive apt-get install -y "$pkg"
else
    pkg=./$dist/tenskit-${version}-1.${rpm_arch}.rpm
    [ -f "$pkg" ] || fail "$pkg is missing"
    listing=$(rpm -qlp "$pkg" 2>/dev/null)
    [ "$(rpm -qp --qf '%{ARCH}' "$pkg" 2>/dev/null)" = "$rpm_arch" ] || fail "wrong architecture in $pkg"
    license=usr/share/licenses/tenskit/LICENSE-MIT
    has_also=usr/share/licenses/tenskit/LICENSE-APACHE
    if command -v dnf >/dev/null 2>&1; then
        dnf install -y "$pkg"
    elif command -v zypper >/dev/null 2>&1; then
        zypper --non-interactive install --allow-unsigned-rpm "$pkg"
    else
        fail "no apt-get, dnf or zypper here"
    fi
fi

# Check the package's file list, not the disk: some container images are set up
# to skip man pages when installing.
has() {
    case $listing in
        *"$1"*) ;;
        *) fail "$pkg does not contain $1" ;;
    esac
}
has "$license"
[ -z "${has_also:-}" ] || has "$has_also"
for tool in $tools; do
    has "usr/bin/$tool"
    has "usr/share/man/man1/$tool.1.gz"

    expected="$tool $version (tenskit.org)"
    actual=$("$tool" --version | head -n 1)
    [ "$actual" = "$expected" ] || fail "$tool --version said '$actual', expected '$expected'"
done

# Do some real work, not only print a version.
tensinfo generated/test-vectors/small.safetensors

echo "OK: $pkg"
