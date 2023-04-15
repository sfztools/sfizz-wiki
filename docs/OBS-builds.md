# Open Build Service

Links to current build channels:
- stable: https://build.opensuse.org/package/show/home:sfztools:sfizz/sfizz
- develop: https://build.opensuse.org/package/show/home:sfztools:sfizz:develop/sfizz

## Maintaining the builds

There are package builds by distribution which are organized as follows:

- RPM-style distribution: build steps in `sfizz.spec`
- Deb-style distribution: build steps in `debian.rules`, metadata in other files `debian.*` and `sfizz.dsc`
- Arch-style distribution: build steps in `PKGBUILD`

This organization is mostly explained in the OBS user manual.

Depend on the target distribution, the package is splitted into multiple subpackages.
- RPM: `sfizz`, `libsfizz0`, `libsfizz0-devel`
- DEB: `sfizz`, `libsfizz0`, `libsfizz0-dev`

The `sfizz` package contains the LV2 and JACK programs. They are statically linked, so they do not have a dependency to `libsfizz0`.

From OBS documentation, it's not clarified how to make Debian package splits.
This is accomplished by specifying multiple `Package` entries in `debian.control`, and listing them again in the `Binary` field of `sfizz.dsc`.

### The service

The OBS packages contain a `_service` file at the root.

The goal of this file is to tell the builder to perform a series of operations before starting to build.
The purpose can be to download the remote source archive, or to set the version of the package which is built.

- [tar_scm](https://github.com/openSUSE/obs-service-tar_scm/blob/master/tar_scm.service.in): This clones the git repo and packs the result into a tarball, which makes it a source file of the build.
   (this service is marked obsolete, but some target platforms don't allow the newer, so it's kept back on purpose)
- [recompress](https://github.com/openSUSE/obs-service-recompress/blob/master/recompress.service): Applies compression on the created tarball to turn it into `tar.gz` or other
- [set_version](https://github.com/openSUSE/open-build-service/blob/master/src/api/test/fixtures/backend/services/set_version.service): Sets version in the package build files. The version number can be explicit, otherwise it can be created based on the git revision.
- [download_url](https://github.com/openSUSE/open-build-service/blob/master/src/api/test/fixtures/backend/services/download_url.service): Download a source file based on URL, and keep it in cache.
- [verify_file](https://github.com/hiberis/obs-service-verify_file/blob/master/verify_file.service): Apply a check based on hash to the downloaded file.

Note: when committing, I can be required to install some OBS services on the local machine.
At least on Arch, they are available in AUR as packages `obs-service-*`.

### Updating the develop channel

The develop channel is triggered by a Github webhook on push. It's fully automatic, and does not need update.

### Updating the stable channel

The stable channel contains a `_service` file that applies a version number automatically into the build files.
To update, it's only needed to edit the `_service` file and none other.

This is the example service of `sfizz 0.3.0`.
To update it's just needed to change the version `0.3.0` everywhere and update the file checksum.

```
<services>
  <service name="set_version">
    <param name="version">0.3.0</param>
  </service>
  <service name="download_url">
    <param name="protocol">https</param>
    <param name="host">github.com</param>
    <param name="path">sfztools/sfizz/releases/download/0.3.0/sfizz-0.3.0-src.tar.gz</param>
  </service>
  <service name="verify_file">
    <param name="file">_service:download_url:sfizz-0.3.0-src.tar.gz</param>
    <param name="verifier">sha256</param>
    <param name="checksum">3ba24c1ca468391b6d2b209c712ea0d5032b7aeaf977f271d06681ce207d95bb</param>
  </service>
</services>
```