"""
Dashboard and Terminal UI Utilities
Provides formatted display, menus, and interactive elements
"""

import os
import sys
from typing import List, Tuple, Any
from datetime import datetime


class Dashboard:
    """Handles terminal UI and display formatting"""
    
    # ANSI Color codes
    COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'END': '\033[0m'
    }
    
    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    @staticmethod
    def print_header(title: str, width: int = 60):
        """Print formatted header"""
        borders = "=" * width
        padding = (width - len(title) - 2) // 2
        print(f"\n{Dashboard.COLORS['BOLD']}{borders}{Dashboard.COLORS['END']}")
        print(f"{' ' * padding} {title}")
        print(f"{Dashboard.COLORS['BOLD']}{borders}{Dashboard.COLORS['END']}\n")
    
    @staticmethod
    def print_menu(title: str, options: List[Tuple[str, str]]) -> str:
        """
        Print interactive menu and get user choice
        
        Args:
            title: Menu title
            options: List of (number, description) tuples
        
        Returns:
            User's choice as string
        """
        Dashboard.clear_screen()
        Dashboard.print_header(title, 70)
        
        for num, desc in options:
            print(f"  {Dashboard.COLORS['CYAN']}{num}{Dashboard.COLORS['END']}) {desc}")
        
        print()
        while True:
            choice = input(f"{Dashboard.COLORS['BOLD']}Enter your choice: {Dashboard.COLORS['END']}").strip()
            valid_choices = [opt[0] for opt in options]
            if choice in valid_choices:
                return choice
            print(f"{Dashboard.COLORS['RED']}❌ Invalid choice. Please try again.{Dashboard.COLORS['END']}")
    
    @staticmethod
    def print_success(message: str):
        """Print success message"""
        print(f"\n{Dashboard.COLORS['GREEN']}✅ {message}{Dashboard.COLORS['END']}")
    
    @staticmethod
    def print_error(message: str):
        """Print error message"""
        print(f"\n{Dashboard.COLORS['RED']}❌ {message}{Dashboard.COLORS['END']}")
    
    @staticmethod
    def print_warning(message: str):
        """Print warning message"""
        print(f"\n{Dashboard.COLORS['YELLOW']}⚠️  {message}{Dashboard.COLORS['END']}")
    
    @staticmethod
    def print_info(message: str):
        """Print info message"""
        print(f"\n{Dashboard.COLORS['BLUE']}ℹ️  {message}{Dashboard.COLORS['END']}")
    
    @staticmethod
    def print_row(cols: List[Any], widths: List[int], align: str = 'left'):
        """Print table row with specified column widths"""
        row = ""
        for col, width in zip(cols, widths):
            col_str = str(col)[:width]
            if align == 'right':
                row += f"{col_str:>{width}} │ "
            else:
                row += f"{col_str:<{width}} │ "
        print(row)
    
    @staticmethod
    def print_table(headers: List[str], rows: List[List[str]], 
                   widths: List[int] = None):
        """Print formatted table"""
        if not rows:
            print(f"\n{Dashboard.COLORS['YELLOW']}No data to display{Dashboard.COLORS['END']}\n")
            return
        
        if widths is None:
            widths = [20] * len(headers)
        
        # Print header
        Dashboard.print_row(headers, widths)
        print("─" * (sum(widths) + len(widths) * 3))
        
        # Print rows
        for row in rows:
            Dashboard.print_row(row, widths)
    
    @staticmethod
    def print_store_profile(store_info: dict):
        """Print store profile information"""
        Dashboard.print_header("📦 STORE PROFILE")
        
        print(f"  {Dashboard.COLORS['BOLD']}Store Name:{Dashboard.COLORS['END']} {store_info.get('store_name', 'N/A')}")
        print(f"  {Dashboard.COLORS['BOLD']}Location:{Dashboard.COLORS['END']} {store_info.get('location', 'N/A')}")
        print(f"  {Dashboard.COLORS['BOLD']}Store Type:{Dashboard.COLORS['END']} {store_info.get('store_type', 'N/A')}")
        print(f"  {Dashboard.COLORS['BOLD']}Investment:{Dashboard.COLORS['END']} ₹{store_info.get('investment', 0):,.0f}")
        print(f"  {Dashboard.COLORS['BOLD']}Created:{Dashboard.COLORS['END']} {store_info.get('created_date', 'N/A')}")
        print(f"  {Dashboard.COLORS['BOLD']}Total Sales:{Dashboard.COLORS['END']} ₹{store_info.get('total_revenue', 0):,.0f}")
        print()
    
    @staticmethod
    def get_input(prompt: str, input_type: str = 'str', 
                  allow_empty: bool = False) -> Any:
        """
        Get user input with validation
        
        Args:
            prompt: Input prompt text
            input_type: Type of input ('str', 'int', 'float', 'date')
            allow_empty: Allow empty input
        
        Returns:
            User input converted to specified type
        """
        while True:
            try:
                user_input = input(f"\n{Dashboard.COLORS['BOLD']}{prompt}{Dashboard.COLORS['END']}").strip()
                
                if not user_input and not allow_empty:
                    Dashboard.print_warning("Input cannot be empty")
                    continue
                
                if not user_input:
                    return None
                
                if input_type == 'int':
                    return int(user_input)
                elif input_type == 'float':
                    return float(user_input)
                elif input_type == 'date':
                    # Parse date in DD-MM-YYYY format
                    datetime.strptime(user_input, "%d-%m-%Y")
                    return user_input
                else:  # str
                    return user_input
            
            except ValueError:
                Dashboard.print_error(f"Invalid input. Please enter a valid {input_type}")
            except Exception as e:
                Dashboard.print_error(f"Error: {str(e)}")
    
    @staticmethod
    def get_yes_no(prompt: str) -> bool:
        """Get Yes/No confirmation from user"""
        while True:
            response = input(f"\n{prompt} (Yes/No): ").strip().lower()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                Dashboard.print_warning("Please enter 'Yes' or 'No'")
    
    @staticmethod
    def print_welcome():
        """Print welcome screen"""
        Dashboard.clear_screen()
        print(f"""
{Dashboard.COLORS['BOLD']}{Dashboard.COLORS['CYAN']}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🏪 AI-BASED SMART GROCERY ACCOUNTING SYSTEM 🏪          ║
║                                                              ║
║          Demand Forecasting & Inventory Optimization         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
{Dashboard.COLORS['END']}
""")
    
    @staticmethod
    def print_divider():
        """Print visual divider"""
        print(f"\n{Dashboard.COLORS['BOLD']}─" * 60 + f"{Dashboard.COLORS['END']}\n")
    
    @staticmethod
    def print_section(title: str):
        """Print section title"""
        print(f"\n{Dashboard.COLORS['BOLD']}{Dashboard.COLORS['BLUE']}{'=' * 60}{Dashboard.COLORS['END']}")
        print(f"{Dashboard.COLORS['BOLD']}{title}{Dashboard.COLORS['END']}")
        print(f"{Dashboard.COLORS['BOLD']}{Dashboard.COLORS['BLUE']}{'=' * 60}{Dashboard.COLORS['END']}\n")
    
    @staticmethod
    def loading_animation(message: str = "Processing", duration: int = 3):
        """Show loading animation"""
        import time
        animations = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        
        for i in range(duration * 4):
            sys.stdout.write(f'\r{animations[i % len(animations)]} {message}...')
            sys.stdout.flush()
            time.sleep(0.25)
        
        print()  # New line after animation
    
    @staticmethod
    def print_statistics(stats: dict):
        """Print statistics dashboard"""
        print(f"\n{Dashboard.COLORS['BOLD']}📊 STATISTICS{Dashboard.COLORS['END']}\n")
        
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        print()
    
    @staticmethod
    def print_alert_box(title: str, message: str, alert_type: str = 'info'):
        """Print alert box"""
        width = max(len(title), len(message)) + 4
        
        colors = {
            'info': Dashboard.COLORS['BLUE'],
            'warning': Dashboard.COLORS['YELLOW'],
            'error': Dashboard.COLORS['RED'],
            'success': Dashboard.COLORS['GREEN']
        }
        
        color = colors.get(alert_type, Dashboard.COLORS['BLUE'])
        
        print(f"\n{color}")
        print("╔" + "═" * (width + 2) + "╗")
        print(f"║ {title:<{width}} ║")
        print("╠" + "═" * (width + 2) + "╣")
        print(f"║ {message:<{width}} ║")
        print("╚" + "═" * (width + 2) + "╝")
        print(f"{Dashboard.COLORS['END']}\n")
