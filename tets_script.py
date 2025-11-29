 #!/usr/bin/env python3

# this is a script made for testing weather all the functions are working correctly or not 
# you might see a sharp spike while executing this wcript expecially the function for indexing code base uses lot of cpu 

import os
import sys
import traceback
from tools import (
    initialize_vectorstore,
    change_directory,
    read_file,
    search_code,
    list_directory,
    run_command,
    add_to_vectorstore,
    search_vectorstore,
    index_codebase
)

def safe_test(func_name, func, *args, **kwargs):
    print(f"\n{'-'*20}")
    print(f"Testing {func_name}")
    print(f"{'-'*10}")

    try:
        result = func(*args, **kwargs)
        print(f" The Function{func_name} is working correctly")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"ERROR detected in {func_name}: {str(e)}")
        print(f"traceback: {traceback.format_exc()}")
        return False

def main():
    print("starting  testing of all tools.py functions")

    # Store original directory
    original_dir = os.getcwd()

    # Test results
    test_results = {}

    # 1. Test initialize_vectorstore (simple function)
    test_results['initialize_vectorstore'] = safe_test(
        'initialize_vectorstore',
        initialize_vectorstore
    )

    # 2. Test list_directory with current directory
    test_results['list_directory'] = safe_test(
        'list_directory',
        list_directory
    )

    # 3. Test change_directory (test with current directory)
    test_results['change_directory'] = safe_test(
        'change_directory',
        change_directory,
        "."
    )

    # 4. Test read_file with this script
    test_results['read_file'] = safe_test(
        'read_file',
        read_file,
        __file__
    )

    # 5. Test search_code (safe grep command)
    test_results['search_code'] = safe_test(
        'search_code',
        search_code,
        "def",
        "."
    )

    # 6. Test run_command (safe command)
    test_results['run_command'] = safe_test(
        'run_command',
        run_command,
        "echo 'Hello World'"
    )
# the main file used for indexing codebases
    print(f"\n{'-'*20}")
    print("Testing add_to_vectorstore (potentially problematic)")
    print(f"{'-'*10}")

    # Create a simple test file first
    test_file_path = "test_content.txt"
    try:
        with open(test_file_path, 'w') as f:
            f.write("lorem ipsum dolor sit amet, consectetur adipiscing elit.")

        test_results['add_to_vectorstore'] = safe_test(
            'add_to_vectorstore',
            add_to_vectorstore,
            test_file_path
        )

        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)

    except Exception as e:
        print(f" Could not create test file for add_to_vectorstore: {e}")
        test_results['add_to_vectorstore'] = False

    if test_results.get('add_to_vectorstore', False):
        test_results['search_vectorstore'] = safe_test(
            'search_vectorstore',
            search_vectorstore,
            "test file"
        )
    else:
        print(f"\n{'-'*20}")
        print("Skipping search_vectorstore (add_to_vectorstore failed)")
        print(f"{'-'*10}")
        test_results['search_vectorstore'] = False

    # 9. Test index_codebase (might be problematic too)
    # We'll test it with a small directory to avoid processing too much
    test_results['index_codebase'] = safe_test(
        'index_codebase',
        index_codebase,
        "."  # Current directory
    )

    # Summary
    print(f"\n{'='*60}")
    print("summary of test results")
    print(f"{'='*60}")

    passed = 0
    failed = 0

    for func_name, result in test_results.items():
        status = "PASS" if result else "FAIL"
        print(f"{func_name:<25}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\nTotal functions tested: {len(test_results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print(f"\n  {failed} function(s) have issues that need to be fixed.")
        sys.exit(1)
    else:
        print(f"\n All functions are working correctly you can proceed further !")
        sys.exit(0)

if __name__ == "__main__":
    main()