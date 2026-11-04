# Installing tenskit

tenskit binaries are available for Linux on 64-bit Intel (x86_64, amd64) and 64-bit ARM (aarch64, arm64). macOS and Windows to follow.

Every method installs the same five tools: `tenspile`, `tensinfo`, `tensdiff`, `tensconv`, and `tensquant`.

| Method | Updates | Needs root |
| --- | --- | --- |
| [Package repository](#install-from-the-package-repository) | With your normal system upgrades | Yes |
| [Package from Releases](#install-a-package-from-releases) | By hand | Yes |
| [Archive from Releases](#install-without-root) | By hand | No |
| [Build from source](#build-from-source) | By hand | No |

Use the package repository if your system is supported and you have root. Otherwise, use the archive: the Linux binaries are statically linked, so it should run on other distributions too.

## Install from the package repository

[packages.goldnode.org](https://packages.goldnode.org/) has signed repositories, so tenskit updates arrive with your normal system upgrades. That page is the current reference, including for supported systems. It covers:

| Distribution | Versions |
| --- | --- |
| Debian | 12, 13 |
| Ubuntu | 22.04, 24.04, 26.04, 26.10 |
| RHEL, Rocky Linux, AlmaLinux, Oracle Linux | 8, 9, 10 |
| Fedora | 44, 45 |
| openSUSE | Leap 16.0, Tumbleweed |

The signing key's fingerprint is:

```
8A87 8DB2 8111 35C2 BC4B  D3EB 3B91 E435 E25D 3DE4
```

### Debian and Ubuntu

```bash
sudo curl -fsSLo /usr/share/keyrings/goldnode-archive-keyring.gpg \
  https://packages.goldnode.org/goldnode-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/goldnode-archive-keyring.gpg] \
https://packages.goldnode.org/deb $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
  | sudo tee /etc/apt/sources.list.d/goldnode.list
sudo apt update
sudo apt install tenskit
```

To update, run `sudo apt update && sudo apt upgrade`.

### RHEL, Rocky Linux, AlmaLinux, and Oracle Linux

```bash
sudo tee /etc/yum.repos.d/goldnode.repo <<EOF
[goldnode]
name=goldnode.org
baseurl=https://packages.goldnode.org/rpm/el/$(. /etc/os-release && echo "${VERSION_ID%%.*}")/\$basearch
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://packages.goldnode.org/gpg.key
EOF
sudo dnf install tenskit
```

dnf asks you to confirm the key on first use. To update, run `sudo dnf upgrade`.

### Fedora

```bash
sudo tee /etc/yum.repos.d/goldnode.repo <<'EOF'
[goldnode]
name=goldnode.org
baseurl=https://packages.goldnode.org/rpm/fedora/$releasever/$basearch
enabled=1
gpgcheck=1
repo_gpgcheck=1
gpgkey=https://packages.goldnode.org/gpg.key
EOF
sudo dnf install tenskit
```

dnf asks you to confirm the key on first use. To update, run `sudo dnf upgrade`.

### openSUSE

Add the repository for your version.

Leap 16.0:

```bash
sudo zypper addrepo --gpgcheck 'https://packages.goldnode.org/rpm/opensuse/$releasever/$basearch' goldnode
```

Tumbleweed:

```bash
sudo zypper addrepo --gpgcheck 'https://packages.goldnode.org/rpm/opensuse/tumbleweed/$basearch' goldnode
```

Then install tenskit:

```bash
sudo zypper refresh
sudo zypper install tenskit
```

zypper asks you to trust the key during the refresh. To update, run `sudo zypper refresh && sudo zypper update tenskit`.

### Remove the repository

Removing the repository doesn't remove tenskit. Uninstall that first if you no longer want it (see [Uninstall](#uninstall)).

```bash
sudo rm /etc/apt/sources.list.d/goldnode.list   # Debian, Ubuntu
sudo rm /etc/yum.repos.d/goldnode.repo          # RHEL and clones, Fedora
sudo zypper removerepo goldnode                 # openSUSE
```

## Install a package from Releases

Packages downloaded from the [Releases](https://github.com/goldnode/tenskit/releases) page don't update by themselves: to update, install the new version's package the same way.

Download the file for your system from the latest release:

| System | x86_64 | aarch64 |
| --- | --- | --- |
| Debian, Ubuntu | `tenskit_VERSION-1_amd64.deb` | `tenskit_VERSION-1_arm64.deb` |
| RHEL and clones, Fedora, openSUSE | `tenskit-VERSION-1.x86_64.rpm` | `tenskit-VERSION-1.aarch64.rpm` |

Run `uname -m` to see which architecture you have. Then install it, shown here for version 0.3.0 on x86_64. The `./` is needed so the tool reads the file instead of searching its repositories.

```bash
# Debian, Ubuntu
sudo apt install ./tenskit_0.3.0-1_amd64.deb

# RHEL and clones, Fedora
sudo dnf install ./tenskit-0.3.0-1.x86_64.rpm

# openSUSE
sudo zypper install --allow-unsigned-rpm ./tenskit-0.3.0-1.x86_64.rpm
```

Packages from Releases aren't GPG-signed, which is why zypper needs `--allow-unsigned-rpm`. To check a download, see [Verify a download](#verify-a-download).

## Install without root

The `.tar.gz` archive from [Releases](https://github.com/goldnode/tenskit/releases) installs into your home directory. Set `version` to the latest release:

```bash
version=0.3.0
arch=$(uname -m)
name=tenskit-$version-$arch-linux

curl -fsSLO "https://github.com/goldnode/tenskit/releases/download/v$version/$name.tar.gz"
tar -xzf "$name.tar.gz"
mkdir -p ~/.local/bin ~/.local/share/man/man1
cp "$name"/bin/* ~/.local/bin/
cp "$name"/share/man/man1/* ~/.local/share/man/man1/
```

If `~/.local/bin` isn't on your `PATH`, add these lines to your shell's startup file, such as `~/.bashrc`. The `MANPATH` line lets `man` find the manual pages; its trailing `:` keeps the system pages available.

```bash
export PATH="$HOME/.local/bin:$PATH"
export MANPATH="$HOME/.local/share/man:"
```

To update, repeat these steps with the new version.

## Build from source

You need [Git](https://git-scm.com/), a recent stable [Rust](https://rustup.rs/) toolchain, and a C compiler such as `gcc`, which Rust uses for linking. tenskit has no other system dependencies.

```bash
git clone https://github.com/goldnode/tenskit.git
cd tenskit
cargo build --release --locked
```

The tools are now in `target/release`, and you can run them from there:

```bash
target/release/tenspile --version
MANPATH=target/release/share/man man tenspile
```

To copy the tools to `~/.cargo/bin`, which the Rust installer normally adds to your `PATH`:

```bash
cargo install --path . --locked
```

`cargo install` doesn't copy the manual pages. To install them too:

```bash
mkdir -p ~/.local/share/man/man1
cp target/release/share/man/man1/* ~/.local/share/man/man1/
```

To update, run `git pull` and repeat these steps. To run the tests, use `cargo test --locked`.

A native build links against your system's C library, so unlike the release binaries, it may not run on other distributions. To build a static binary like the released ones:

```bash
rustup target add x86_64-unknown-linux-musl   # or aarch64-unknown-linux-musl
cargo build --release --locked --target x86_64-unknown-linux-musl
```

The tools are then in `target/x86_64-unknown-linux-musl/release`.

## Verify a download

Each release has a `SHA256SUMS` file. Download it to the same directory as your file, then run:

```bash
sha256sum -c SHA256SUMS --ignore-missing
```

Releases also have build provenance, which shows the file was built by this repository's GitHub workflow. To check it, use the [GitHub CLI](https://cli.github.com/):

```bash
gh attestation verify tenskit_0.3.0-1_amd64.deb --repo goldnode/tenskit
```

## Check the install

```bash
tenspile --version
man tenspile
```

Some minimal systems and containers leave out `man` or skip installing manual pages. The tools still work, and each takes `-h` for a short help.

Then try the [Getting started](getting-started.md) walk-through.

## Uninstall

```bash
sudo apt remove tenskit      # Debian, Ubuntu
sudo dnf remove tenskit      # RHEL and clones, Fedora
sudo zypper remove tenskit   # openSUSE
```

For an archive install:

```bash
cd ~/.local/bin && rm -f tenspile tensinfo tensdiff tensconv tensquant
cd ~/.local/share/man/man1 && rm -f tenspile.1.gz tensinfo.1.gz tensdiff.1.gz tensconv.1.gz tensquant.1.gz
```

For `cargo install`, run `cargo uninstall tenskit`.

## macOS and Windows

macOS and Windows to follow. There are no packages for them yet.
