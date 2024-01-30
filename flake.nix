# good nix template: https://srid.ca/rust-nix

{
inputs = {
  nixpkgs.url = "nixpkgs";
  rust-overlay.url = "github:oxalica/rust-overlay";
  flake-utils.url = "github:numtide/flake-utils";
};

outputs = { self, nixpkgs, rust-overlay, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system:
  let
    inherit (nixpkgs) lib;

    overlays = [ (import rust-overlay) ];
    pkgs = import nixpkgs {
      inherit system overlays;
    };
    # pkgs = nixpkgs.legacyPackages.${system};

    # The reason for using pkgs.symlinkJoin instead of just pkgs is to consolidate these various Rust-related components into a single symlink. This can be convenient for setting up a development environment or ensuring that specific tools are available in a unified location. It simplifies the management of Rust-related tools and makes it easier to reference them in the rest of the Nix configuration, for example, in the subsequent nativeBuildInputs section of the mkShell environment.
    # rust-toolchain = pkgs.symlinkJoin {
    #     name = "rust-toolchain";
    #     paths = [ pkgs.rustc-wasm32 pkgs.cargo pkgs.cargo-watch pkgs.rust-analyzer pkgs.rustPlatform.rustcSrc ];
    # };

    rust-toolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
    llvm = pkgs.llvmPackages_16;

    clangBuildInputs = with llvm; [
      clang
      libclang
      libcxx
      libcxxabi
      lld
      lldb
    ];

    nativeBuildInputs = with pkgs; [
      pkg-config
      rust-toolchain
      alsa-lib
      cmake
    ] ++ clangBuildInputs;

    packages = with pkgs; [
      curl
      wget
      dbus
      openssl_3
      glib
      gtk3
      gdk-pixbuf
      libsoup
      pango
      harfbuzz
      at-spi2-atk
      cairo
      webkitgtk
      webkitgtk_4_1
      librsvg
      libayatana-appindicator
      nodejs_18
      nodePackages.pnpm
    ];

    libraries = with pkgs;[
      webkitgtk
      webkitgtk_4_1
      gtk3
      libayatana-appindicator
    ];

    buildInputs = packages;
  in
  rec {
    # `nix develop`
    devShell = pkgs.mkShell {
      inherit buildInputs nativeBuildInputs;
      shellHook = ''
      # make sure all libraries are added into LD_LIBRARY_PATH
      export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath libraries}:$LD_LIBRARY_PATH

      # For rust-analyzer 'hover' tooltips to work.
      export RUST_SRC_PATH=${rust-toolchain}
      export RUST_BACKTRACE=1

      # add ~/.cargo/bin to PATH for crates installed with `cargo install`
      export PATH=$PATH:$HOME/.cargo/bin

      # clang
      export LIBCLANG_PATH="${llvm.libclang.lib}/lib";

      # tauri
      export WEBKIT_DISABLE_COMPOSITING_MODE=1
      '';
    };
  });
}
