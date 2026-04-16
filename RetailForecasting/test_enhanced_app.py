#!/usr/bin/env python3
"""
Quick Test Script - Verify Enhanced App Components
Run this to verify all modules are working correctly
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_imports():
    """Test that all modules import successfully"""
    print("\n🧪 Testing Module Imports...\n")
    
    modules = {
        'UserManager': 'users.user_manager',
        'ProductManager': 'users.product_manager',
        'Dashboard': 'interface.dashboard',
        'InputHandler': 'interface.input_handler',
        'SalesAnalytics': 'analytics.sales_analytics',
        'InventoryManager': 'inventory.inventory_manager',
    }
    
    passed = 0
    failed = 0
    
    for name, module_path in modules.items():
        try:
            exec(f"from {module_path} import {name}")
            print(f"✅ {name:20s} ({module_path})")
            passed += 1
        except Exception as e:
            print(f"❌ {name:20s} - {str(e)}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests Passed: {passed}/{len(modules)}")
    print(f"{'='*60}\n")
    
    return failed == 0

def test_data_structure():
    """Test that data directory structure is set up"""
    print("\n🧪 Checking Data Structure...\n")
    
    required_dirs = [
        'data',
        'data/user',
        'src/users',
        'src/interface',
        'src/analytics',
        'src/inventory'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - NOT FOUND")
            all_exist = False
    
    print(f"\n{'='*60}")
    if all_exist:
        print("✅ All directories present")
    else:
        print("⚠️  Some directories are missing")
    print(f"{'='*60}\n")
    
    return all_exist

def test_required_files():
    """Test that required files exist"""
    print("\n🧪 Checking Required Files...\n")
    
    required_files = [
        'app_enhanced.py',
        'src/users/user_manager.py',
        'src/users/product_manager.py',
        'src/interface/dashboard.py',
        'src/interface/input_handler.py',
        'src/analytics/sales_analytics.py',
        'src/inventory/inventory_manager.py',
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path:50s} ({size:6d} bytes)")
        else:
            print(f"❌ {file_path:50s} - NOT FOUND")
            all_exist = False
    
    print(f"\n{'='*60}")
    if all_exist:
        print("✅ All files present")
    else:
        print("⚠️  Some files are missing")
    print(f"{'='*60}\n")
    
    return all_exist

def show_quick_start():
    """Show quick start instructions"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║       🎉 ENHANCED SMART GROCERY APP - READY TO USE! 🎉    ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝

📚 QUICK START GUIDE:

1️⃣  ENSURE CONDA ENVIRONMENT IS ACTIVE:
   $ conda activate sales_pred

2️⃣  RUN THE ENHANCED APP:
   $ python app_enhanced.py

3️⃣  FOLLOW THE INTERACTIVE MENUS:
   ├── Register new store (first time)
   ├── Select existing store (returning)
   └── Navigate through features

4️⃣  KEY FEATURES:
   ✅ Daily Sales Entry
   ✅ Sales Analytics
   ✅ Inventory Management
   ✅ Product Management
   ✅ Monthly Predictions
   ✅ Store Profile

📖 DOCUMENTATION:
   • SYSTEM_COMPLETE_GUIDE.md        - Complete overview
   • ENHANCED_APP_GUIDE.md           - Feature details
   • CONDENSED_ARCHITECTURE.md       - System design
   • CONDA_SETUP.md                  - Environment setup

🔗 SUPPORTED APPLICATIONS:
   • app_enhanced.py    - New interactive app (RECOMMENDED)
   • main.py            - Original CLI menu
   • app/app.py         - Streamlit web dashboard

💡 EXAMPLE WORKFLOW (Day 1):
   1. Register store: "Raj's Grocery"
   2. Location: Urban
   3. Type: Medium
   4. Investment: ₹100,000
   5. Initialize 20 suggested products
   6. Record 5 daily sales
   7. View analytics dashboard
   8. Check inventory recommendations

⏱️  EXPECTED TIME: 5 minutes to get started

═════════════════════════════════════════════════════════════

Ready to launch? Run:
   $ conda activate sales_pred
   $ python app_enhanced.py

Happy Forecasting! 📊🚀
""")

def main():
    """Run all tests"""
    print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║     ENHANCED SMART GROCERY APP - SYSTEM VERIFICATION      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")
    
    # Run tests
    imports_ok = test_imports()
    dirs_ok = test_data_structure()
    files_ok = test_required_files()
    
    # Summary
    print("\n" + "═"*60)
    print("VERIFICATION SUMMARY")
    print("═"*60 + "\n")
    
    if imports_ok and dirs_ok and files_ok:
        print("🎉 ALL SYSTEMS GO!")
        print("\n✅ Imports: OK")
        print("✅ Directories: OK")
        print("✅ Files: OK")
        print("\n🚀 You can now run the enhanced app:\n")
        print("   $ python app_enhanced.py\n")
    else:
        print("⚠️  SOME ISSUES FOUND:")
        if not imports_ok:
            print("   ❌ Module import issues - check Python path")
        if not dirs_ok:
            print("   ❌ Missing directories - ensure correct location")
        if not files_ok:
            print("   ❌ Missing files - verify download/creation")
        print("\n📖 For help, see SYSTEM_COMPLETE_GUIDE.md\n")
    
    # Show quick start
    show_quick_start()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during verification: {str(e)}\n")
        sys.exit(1)
