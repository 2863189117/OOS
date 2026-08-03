#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
lock="$root/deps.lock.json"
lock_hash="$(cmake -E sha256sum "$lock" | awk '{print $1}')"
base="${OOS_DEPS_ROOT:-$root/.deps/$lock_hash}"
downloads="$base/downloads"
sources="$base/sources"
builds="$base/builds"
prefix="$base/install"
mkdir -p "$downloads" "$sources" "$builds" "$prefix"

read_lock() {
  cmake -DLOCK="$lock" -DNAME="$1" -P "$root/cmake/read_lock.cmake"
}

fetch_unpack() {
  local name="$1" fields version url sha archive source
  fields="$(read_lock "$name")"
  version="$(sed -n 's/^version=//p' <<<"$fields")"
  url="$(sed -n 's/^url=//p' <<<"$fields")"
  sha="$(sed -n 's/^sha256=//p' <<<"$fields")"
  archive="$downloads/$name-$version.tar.gz"
  source="$sources/$name-$version"
  if [[ ! -f "$archive" ]] || [[ "$(cmake -E sha256sum "$archive" | awk '{print $1}')" != "$sha" ]]; then
    curl -L --fail --retry 3 "$url" -o "$archive.tmp"
    if [[ "$(cmake -E sha256sum "$archive.tmp" | awk '{print $1}')" != "$sha" ]]; then
      echo "checksum mismatch for $name $version" >&2
      exit 1
    fi
    mv "$archive.tmp" "$archive"
  fi
  if [[ ! -f "$source/.unpacked" ]]; then
    rm -rf "$source"
    mkdir -p "$source"
    tar -xzf "$archive" --strip-components=1 -C "$source"
    touch "$source/.unpacked"
  fi
  printf '%s\n' "$source"
}

build_dep() {
  local name="$1" source="$2"; shift 2
  local build="$builds/$name"
  cmake -S "$source" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON "$@"
  cmake --build "$build" --parallel "${OOS_BUILD_JOBS:-24}"
  cmake --install "$build"
}

build_dep embree "$(fetch_unpack embree)" \
  -DEMBREE_TASKING_SYSTEM=INTERNAL -DEMBREE_ISPC_SUPPORT=OFF \
  -DEMBREE_TUTORIALS=OFF -DEMBREE_TESTING=OFF
build_dep hdf5 "$(fetch_unpack hdf5)" \
  -DHDF5_BUILD_CPP_LIB=OFF -DHDF5_BUILD_FORTRAN=OFF \
  -DHDF5_BUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
build_dep yaml-cpp "$(fetch_unpack yaml-cpp)" \
  -DYAML_CPP_BUILD_TESTS=OFF -DYAML_CPP_BUILD_TOOLS=OFF
build_dep nlohmann_json "$(fetch_unpack nlohmann_json)" \
  -DJSON_BuildTests=OFF
build_dep Catch2 "$(fetch_unpack catch2)" \
  -DCATCH_BUILD_TESTING=OFF -DCATCH_INSTALL_DOCS=OFF

cat >"$base/environment.sh" <<EOF
export OOS_DEPS_PREFIX="$prefix"
export CMAKE_PREFIX_PATH="$prefix"
EOF
printf 'dependency prefix: %s\nlock hash: %s\n' "$prefix" "$lock_hash"
