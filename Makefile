ROOT_DIR := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))
NPROCS := $(shell echo $$(nproc))
LLVM_LIT := $(shell venv/bin/python3 -c "import pkg_resources; print(pkg_resources.get_distribution('lit').location+'/../../../bin/lit')")
LLVM_BIN_DIR := $(shell venv/bin/python3 -c "import lingodbllvm; print(lingodbllvm.get_bin_dir())")

build:
	mkdir -p $@
venv:
	python3 -m venv venv
	venv/bin/pip install -r requirements.txt
	venv/bin/python3 -c "import pyarrow; pyarrow.create_library_symlinks()"

LDB_ARGS= -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		 -DPython3_EXECUTABLE="venv/bin/python3" \
	   	 -DCMAKE_BUILD_TYPE=Debug

build/dependencies: venv build
	touch $@

build/chase-debug/.stamp: build/dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS)
	touch $@

build/chase-debug/.buildstamp: build/chase-debug/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@


build/chase-release/.buildstamp: build/chase-release/.stamp
	cmake --build $(dir $@) -- -j${NPROCS}
	touch $@

build/chase-release/.stamp: build/dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS_RELEASE="-DDEBUG"
	touch $@
build/chase-debug-coverage/.stamp: build/dependencies
	cmake -G Ninja . -B $(dir $@) $(LDB_ARGS) -DCMAKE_CXX_FLAGS=--coverage -DCMAKE_C_FLAGS=--coverage
	touch $@

.PHONY: run-test
run-test: build/chase-debug/.stamp
	cmake --build $(dir $<) --target mlir-db-opt run-mlir run-sql sql-to-mlir -- -j${NPROCS}
	$(MAKE) test-no-rebuild

coverage: build/chase-debug-coverage/.stamp
	$(MAKE) test-coverage
	genhtml  --ignore-errors source $(dir $<)/filtered-coverage.info --legend --title "lcov-test" --output-directory=$(dir $<)/coverage-report
	open $(dir $<)/coverage-report/index.html


build-docker-dev:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t chase-dev --target baseimg "."

build-docker-py-dev:
	DOCKER_BUILDKIT=1 docker build -f "tools/python/bridge/Dockerfile" -t chase-py-dev --target devimg "."
build-py-bridge:
	DOCKER_BUILDKIT=1 docker build -f "tools/python/bridge/Dockerfile" -t chase-py-dev-build --target build "."
	docker run --rm  -v "${ROOT_DIR}:/built-packages" chase-py-dev-build create_package.sh cp$(PY_VERSION)-cp$(PY_VERSION)

build-docker:
	DOCKER_BUILDKIT=1 docker build -f "tools/docker/Dockerfile" -t lingo-db:latest --target chase  "."

build-release: build/chase-release/.buildstamp
build-debug: build/chase-debug/.buildstamp

.repr-docker-built:
	$(MAKE) build-repr-docker
	touch .repr-docker-built

.PHONY: clean
clean:
	rm -rf build

lint: build/chase-debug/.stamp
	cmake --build build/chase-debug --target build_includes
	sed -i 's/-fno-lifetime-dse//g' build/chase-debug/compile_commands.json
	venv/bin/python3 tools/scripts/run-clang-tidy.py -p $(dir $<) -quiet -header-filter="$(shell pwd)/include/.*" -exclude="arrow|vendored" -clang-tidy-binary=${LLVM_BIN_DIR}/clang-tidy
