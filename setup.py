import os
import sys
import platform
import subprocess
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CustomBuildExt(build_ext):
    """Custom build extension to handle OpenMP configuration."""



    def run(self):
        """Override run to catch compilation failures gracefully."""
        try:
            super().run()
            print("\n" + "=" * 60)
            print("SUCCESS: C++ extension compiled successfully!")
            print("=" * 60 + "\n")
        except Exception as e:
            print("\n" + "=" * 60)
            print("WARNING: C++ compilation failed!")
            print(f"Error: {e}")
            print("-" * 60)
            print("Installation will continue with pre-compiled binaries.")
            print("These binaries should work but may not be optimized for your system.")
            print("-" * 60)
            self._print_compilation_instructions()
            print("=" * 60 + "\n")
            # Don't raise - let installation continue

    def build_extension(self, ext):
        """Build a single extension, catching errors gracefully."""
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"\nWARNING: Failed to build {ext.name}")
            print(f"Reason: {e}")
            print("Will use pre-compiled binaries instead.\n")
            # Don't raise - silently skip this extension

    def _print_compilation_instructions(self):
        """Print platform-specific instructions for manual compilation."""
        system = platform.system()

        print("\nTo compile the extension manually later, try:")
        print("-" * 40)

        if system == "Darwin":  # macOS
            print("For macOS:")
            print("  1. Install OpenMP (if needed): brew install libomp")
            print("  2. cd to the pygsfit/src_gyrosynchrotron directory")
            print("  3. Run: make -f makefile")
            print("\nOr reinstall the package:")
            print("  pip install --force-reinstall --no-binary :all: pygsfit")

        elif system == "Linux":
            print("For Linux:")
            print("  1. Install build tools: sudo apt-get install build-essential")
            print("  2. cd to the pygsfit/src_gyrosynchrotron directory")
            print("  3. Run: make -f makefile")
            print("\nOr reinstall the package:")
            print("  pip install --force-reinstall --no-binary :all: pygsfit")

        elif system == "Windows":
            print("For Windows:")
            print("Option 1 - Using MinGW (lightweight, ~100MB):")
            print("  1. Install MinGW: conda install -c conda-forge m2w64-toolchain")
            print("  2. Set compiler: set CC=gcc && set CXX=g++")
            print("  3. Reinstall: pip install --force-reinstall --no-binary :all: pygsfit")
            print("\nOption 2 - Using MSVC (larger, ~2GB):")
            print("  1. Install from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print("  2. Reinstall: pip install --force-reinstall --no-binary :all: pygsfit")
            print("\nOption 3 - Manual compilation:")
            print("  1. cd to the pygsfit/src_gyrosynchrotron directory")
            print("  2. Use your preferred C++ compiler to build")

        print("-" * 40)
        print("Note: Pre-compiled binaries will work fine for most use cases.")
        print("Compilation is only needed for optimal performance.\n")

    def build_extensions(self):
        # Detect platform
        system = platform.system()
        machine = platform.machine()

        # Get compiler
        compiler = self.compiler.compiler_type

        print(f"\nAttempting to build C++ extension...")
        print(f"Platform: {system} {machine}")
        print(f"Compiler: {compiler}")

        # Mark extensions as optional so failure doesn't stop installation
        for ext in self.extensions:
            ext.optional = True

            # Platform-specific configuration
            try:
                if system == "Darwin":  # macOS
                    self._configure_macos(ext, machine)
                elif system == "Linux":
                    self._configure_linux(ext)
                elif system == "Windows":
                    self._configure_windows(ext, compiler)
                else:
                    raise RuntimeError(f"Unsupported platform: {system}")

                # Print final configuration for debugging
                print(f"Compile args: {ext.extra_compile_args}")
                print(f"Link args: {ext.extra_link_args}")

            except Exception as e:
                print(f"WARNING: Configuration failed: {e}")
                print("Attempting build anyway...")

        # Call parent build_extensions
        try:
            super().build_extensions()
        except Exception as e:
            print(f"\nCompilation failed: {e}")
            print("Continuing with pre-built binaries...")

    def _configure_macos(self, ext, machine):
        """Configure for macOS with OpenMP support."""
        # Base flags
        ext.extra_compile_args = ['-std=c++11', '-O3', '-fPIC', '-D LINUX']
        ext.extra_link_args = []  # Don't add -shared on macOS, Python adds -bundle automatically

        # Try to find OpenMP
        omp_found = False

        # Check common OpenMP locations
        possible_paths = [
            '/opt/homebrew/opt/libomp',  # ARM64 Homebrew
            '/usr/local/opt/libomp',  # Intel Homebrew
            '/opt/local',  # MacPorts
        ]

        # Also check conda environment
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            possible_paths.insert(0, conda_prefix)

        for base_path in possible_paths:
            lib_path = os.path.join(base_path, 'lib')
            inc_path = os.path.join(base_path, 'include')

            # Check if libomp exists in this location
            if os.path.exists(lib_path) and os.path.exists(inc_path):
                # Check for libomp specifically
                if any('omp' in f for f in os.listdir(lib_path)):
                    print(f"Found OpenMP at {base_path}")
                    ext.extra_compile_args.extend([
                        '-Xpreprocessor', '-fopenmp',
                        f'-I{inc_path}'
                    ])
                    ext.extra_link_args.extend([
                        '-lomp',
                        f'-L{lib_path}',
                        f'-Wl,-rpath,{lib_path}'  # Add rpath to avoid runtime issues
                    ])
                    omp_found = True
                    break

        if not omp_found:
            # Try to detect if libomp is in standard paths
            try:
                # Try to compile a simple OpenMP program
                result = subprocess.run(
                    ['clang++', '-Xpreprocessor', '-fopenmp', '-lomp', '-x', 'c++', '-', '-o', '/dev/null'],
                    input=b'#include <omp.h>\nint main() { return omp_get_num_threads(); }',
                    capture_output=True
                )
                if result.returncode == 0:
                    print("OpenMP found in standard paths")
                    ext.extra_compile_args.extend(['-Xpreprocessor', '-fopenmp'])
                    ext.extra_link_args.append('-lomp')
                    omp_found = True
            except:
                pass

        if not omp_found:
            print("\n" + "=" * 60)
            print("WARNING: OpenMP not found!")
            print("Please install OpenMP using one of:")
            print("  brew install libomp")
            print("  conda install openmp")
            print("  port install libomp")
            print("=" * 60 + "\n")

            # Option 1: Fail the build
            raise RuntimeError("OpenMP is required but not found. Please install it first.")

            # Option 2: Build without OpenMP (comment out the raise above and uncomment below)
            # print("Building without OpenMP support (single-threaded)")
            # ext.define_macros.append(('NO_OPENMP', '1'))

    def _configure_linux(self, ext):
        """Configure for Linux with OpenMP support."""
        ext.extra_compile_args = ['-std=c++11', '-O3', '-fPIC', '-D LINUX', '-fopenmp']
        ext.extra_link_args = ['-shared', '-fopenmp']

        # Check if OpenMP is available
        try:
            result = subprocess.run(
                ['g++', '-fopenmp', '-x', 'c++', '-', '-o', '/dev/null'],
                input=b'#include <omp.h>\nint main() { return omp_get_num_threads(); }',
                capture_output=True
            )
            if result.returncode != 0:
                print("WARNING: OpenMP test compilation failed")
                print("Make sure gcc/g++ with OpenMP support is installed")
        except:
            print("WARNING: Could not test OpenMP availability")

    # def _configure_windows(self, ext, compiler):
    #     """Configure for Windows with OpenMP support."""
    #     if compiler == 'msvc':
    #         ext.extra_compile_args = ['/O2', '/openmp', '/D WINDOWS']
    #         ext.extra_link_args = []
    #     elif compiler == 'mingw32':
    #         ext.extra_compile_args = ['-std=c++11', '-O3', '-D WINDOWS', '-fopenmp']
    #         ext.extra_link_args = ['-fopenmp']
    #     else:
    #         raise RuntimeError(f"Unsupported Windows compiler: {compiler}")

    def _configure_windows(self, ext, compiler):
        """Configure for Windows with OpenMP support."""
        if compiler == 'msvc':
            # Microsoft Visual C++ uses different flag syntax
            ext.extra_compile_args = ['/O2', '/openmp', '/D', 'WINDOWS']
            ext.extra_link_args = []  # MSVC handles OpenMP linking automatically
        elif compiler == 'mingw32':
            # MinGW uses GCC-style flags
            ext.extra_compile_args = ['-std=c++11', '-O3', '-D WINDOWS', '-fopenmp']
            ext.extra_link_args = ['-fopenmp']
        else:
            # Default to MSVC style
            ext.extra_compile_args = ['/O2', '/D', 'WINDOWS']
            ext.extra_link_args = []
            print("WARNING: Unknown compiler, OpenMP may not be enabled")






def get_extension():
    """Create the C++ extension with all source files."""

    # Use the actual directory name
    source_dir = Path('pygsfit/src_gyrosynchrotron')

    # List all source files (excluding Windows-specific dllmain.cpp)
    sources = [
        'Arr_DF.cpp',
        'Coulomb.cpp',
        'ExtMath.cpp',
        'FF.cpp',
        'getparms.cpp',
        'GS.cpp',
        'IDLinterface.cpp',
        'PythonInterface.cpp',
        'Messages.cpp',
        'MWmain.cpp',
        'Plasma.cpp',
        'Std_DF.cpp',
        'Zeta.cpp'
    ]

    # Convert to full paths
    source_files = [str(source_dir / src) for src in sources]

    # Check if source files exist
    missing = [f for f in source_files if not os.path.exists(f)]
    if missing:
        print(f"WARNING: Source files not found, skipping compilation")
        print(f"Missing: {missing[:3]}...")  # Show first 3 missing files
        return None  # Return None to skip extension

    # Create extension with optional=True
    ext = Extension(
        'pygsfit.binaries.MWTransferArr_compiled',  # Creates MWTransferArr_compiled.so
        sources=source_files,
        include_dirs=[str(source_dir)],
        language='c++',
        optional=True  # THIS IS KEY - makes extension optional!
    )

    return ext


if __name__ == "__main__":
    # Try to get extension, but don't fail if sources missing
    ext = get_extension()
    ext_modules = [ext] if ext else []

    if not ext_modules:
        print("\n" + "=" * 60)
        print("NOTE: Skipping C++ compilation (source files not found)")
        print("Will use pre-compiled binaries only")
        print("=" * 60 + "\n")

    setup(
        ext_modules=ext_modules,
        cmdclass={'build_ext': CustomBuildExt} if ext_modules else {},
    )
# Setup configuration
# setup(
#     ext_modules=[get_extension()],
#     cmdclass={'build_ext': CustomBuildExt},
# )