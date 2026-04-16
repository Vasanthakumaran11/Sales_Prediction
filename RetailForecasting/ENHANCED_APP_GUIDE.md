# 🚀 Enhanced Smart Grocery AI System - User Guide

## Overview

The **Enhanced Smart Grocery AI System** is a professional-grade terminal application that helps grocery store owners manage inventory, track sales, and forecast demand using machine learning.

## Features

### 1. **Store Registration & Management**
- Register new stores with location, type, and investment information
- Automatic product suggestions based on investment level
- Store profiles with performance tracking

### 2. **Intelligent Product Management**
- 40+ product categories
- Budget-based product recommendations
- Dynamic inventory initialization
- Stock level tracking

### 3. **Daily Sales Tracking**
- Record sales with multiple parameters:
  - Date (auto or manual)
  - Product selection
  - Quantity and pricing
  - Discounts and promotions
  - Holiday and special conditions
- Automatic revenue calculation

### 4. **Advanced Inventory Management**
- Safety stock calculations (Z-score method)
- Reorder point calculations
- Economic Order Quantity (EOQ) computation
- Risk level assessment (LOW/MEDIUM/HIGH/CRITICAL)
- Stockout probability analysis

### 5. **Sales Analytics**
- Comprehensive sales performance metrics
- Product-wise performance tracking
- Promotional impact analysis
- Daily averages and trends
- Monthly summaries

### 6. **Demand Forecasting** (ML Integration)
- Personalized demand predictions
- Seasonal adjustment
- Festival impact calculation
- Promotion effect modeling

## Getting Started

### Installation

```bash
# Navigate to project directory
cd RetailForecasting

# Ensure conda environment is active
conda activate sales_pred

# Run the enhanced app
python app_enhanced.py
```

### First Time - New Store Registration

1. **Start the application**
   ```bash
   python app_enhanced.py
   ```

2. **Select "New Grocery (First-time user)"**
   - Enter store name (e.g., "Raj's Grocery")
   - Choose location (Urban/Semi-Urban/Rural)
   - Select store type (Small/Medium/Supermarket)
   - Enter initial investment amount (₹10,000 minimum)

3. **View Product Suggestions**
   The system will suggest products based on your investment:
   - **Budget (< ₹50K)**: Essential items
   - **Moderate (₹50K-₹150K)**: Added snacks and beverages
   - **Premium (₹150K-₹500K)**: Full range including premium items
   - **Enterprise (> ₹500K)**: Complete supermarket inventory

4. **Initialize Products**
   - Choose to auto-initialize with suggested products
   - Or manually add products later

## Main Menu Operations

### 1. Daily Sales Entry

```
Menu: 1) Daily Sales Entry
```

**Process:**
1. Enter sale date (or use today's date)
2. Select product from list
3. Enter units sold and unit price
4. Apply discount (if any)
5. Confirm promotional sale status
6. Mark if it's a holiday
7. Indicate if shop was closed

**Output:**
- Sale ID generated
- Revenue automatically calculated
- Store statistics updated
- Data persisted to CSV

**Example:**
```
Date: 15-04-2026
Product: Milk
Units: 20
Price: ₹28
Discount: 5%
Promo: Yes
Holiday: No
Shop Closed: No

✅ Sale recorded! Revenue: ₹532
```

### 2. View Sales History & Analytics

```
Menu: 2) View Sales History & Analytics
```

**Displays:**
- **Overall Metrics**
  - Total revenue
  - Total units sold
  - Number of products

- **Daily Averages**
  - Average daily revenue
  - Average daily units sold
  - Active selling days

- **Product Performance**
  - Units sold per product
  - Revenue per product
  - Average transaction price
  - Transaction count

- **Insights**
  - Best-selling product
  - Promotion effectiveness
  - Market trends

**Example Output:**
```
📊 SALES ANALYTICS SUMMARY
==================================================

Overall Metrics:
  Total Revenue: ₹15,240.00
  Total Units Sold: 215 units
  Total Products: 8

Daily Averages:
  Avg Daily Revenue: ₹1,900.00
  Avg Daily Units: 26.9 units
  Active Days: 8 days

Best Performer:
  Product: Milk
  Units Sold: 65 units

Promotional Impact:
  Promo Boost: 25.3%
  Promo Transactions: 12
```

### 3. Inventory Management

```
Menu: 3) Inventory Management
```

**Workflow:**
1. Select a product from list
2. View inventory analysis
3. Get recommendations

**Recommendation Includes:**
- **Current Status**
  - Current stock
  - Average daily demand
  - Days of supply available

- **Risk Assessment**
  - 🟢 LOW: Safe inventory levels
  - 🟡 MEDIUM: Normal operations
  - 🔴 HIGH: Reorder recommended
  - 🔴🔴 CRITICAL: Urgent action needed

- **Inventory Metrics**
  - Safety stock (Z × σ × √LeadTime)
  - Reorder point
  - Economic Order Quantity (EOQ)
  - Stockout probability

- **Recommendations**
  - Order action (NO ACTION/PRIORITY/URGENT)
  - Suggested order quantity
  - Daily holding cost

**Example:**
```
📦 INVENTORY RECOMMENDATION: Milk
==================================================

Current Status:
  Current Stock: 45 units
  Avg Daily Demand: 8.5 units/day
  Days of Supply: 5.3 days

Risk Level: 🟡 MEDIUM
  Stockout Risk: 35.2%

Recommendations:
  Safety Stock: 5 units
  Reorder Point: 22 units
  Economic Order Qty: 85 units

Order Action:
  PRIORITY REORDER
  Suggested Order Quantity: 85 units
```

### 4. Product Management

```
Menu: 4) Product Management
```

**Sub-options:**
1. **View All Products**
   - Display all inventory
   - Product details
   - Current stock levels
   - Pricing

2. **Add New Product**
   - Name
   - Category
   - Initial stock
   - Unit price

3. **Update Stock**
   - Select product
   - Enter new quantity
   - Automatic timestamp

**Example:**
```
📦 ALL PRODUCTS
========================================================

Product Name | Category | Stock | Price | Last Updated
─────────────────────────────────────────────────────
Milk | Perishables | 45 | ₹28.00 | 15-04-2026
Curd | Perishables | 32 | ₹45.00 | 15-04-2026
Bread | Perishables | 28 | ₹35.00 | 15-04-2026
Rice | Non-Perishables | 120 | ₹65.00 | 14-04-2026
```

### 5. Monthly Prediction

```
Menu: 5) Monthly Prediction
```

**Requirements:**
- Minimum 14 days of sales data
- At least 20+ transactions

**Provides:**
- Month-wise demand forecast
- Festival impact adjustment
- Promotion effect estimation
- Demand change percentage

**Output Format:**
```
🔮 MONTHLY PREDICTION - May 2026
==================================================

Base Forecast:
  Expected Daily Demand: 32 units
  Monthly Total: 960 units
  Expected Revenue: ₹26,880

Adjustments:
  Festival Impact: +15% (960 → 1,104)
  Promotion Impact: +20% (1,104 → 1,325)

Final Forecast:
  Recommended Order: 1,400 units
  Safety Stock: 25 units
  Expected Revenue: ₹37,100
```

### 6. View Store Profile

```
Menu: 6) View Store Profile
```

**Displays:**
- Store name
- Location and type
- Investment amount
- Registration date
- Total sales volume
- Total revenue

## Data Storage Structure

```
data/user/
└── <store_name>/
    ├── profile.json          # Store metadata
    ├── products.csv          # Product inventory
    ├── sales.csv             # Sales transactions
    ├── purchases.csv         # Purchase records
    └── dataset.csv           # ML training data
```

### File Formats

**profile.json:**
```json
{
  "store_name": "Raj's Grocery",
  "location": "Urban",
  "store_type": "Medium",
  "investment": 150000,
  "created_date": "15-04-2026 10:30:45",
  "last_accessed": "15-04-2026 14:22:15",
  "total_sales": 215,
  "total_revenue": 15240.00
}
```

**sales.csv:**
```
Sale_ID,Date,Product_Name,Units_Sold,Unit_Price,Discount,Revenue,Promo,Holiday,Shop_Closed
SALE_1713176445,15-04-2026,Milk,20,28.00,5,532.00,0,0,0
SALE_1713176470,15-04-2026,Rice,15,65.00,0,975.00,0,0,0
```

## Investment Categories & Product Recommendations

### Budget (< ₹50,000)
- **Recommended Products**: 12-15
- **Categories**: Perishables, Non-Perishables
- **Focus**: Essential daily-use items
- **Examples**: Milk, Rice, Oil, Sugar, Bread

### Moderate (₹50,000 - ₹150,000)
- **Recommended Products**: 20-25
- **Categories**: + Snacks & Biscuits, Beverages
- **Focus**: Daily necessities + variety
- **Initial Inventory**: ₹80,000

### Premium (₹150,000 - ₹500,000)
- **Recommended Products**: 30-40
- **Categories**: + Frozen Foods, Personal Care
- **Focus**: Full range grocery store
- **Initial Inventory**: ₹300,000

### Enterprise (> ₹500,000)
- **Recommended Products**: 50+
- **Categories**: All available categories
- **Focus**: Complete supermarket
- **Initial Inventory**: ₹1,000,000

## Inventory Calculation Formulas

### Safety Stock
```
Safety Stock = Z × σ × √LeadTime

Where:
  Z = Service level (1.96 for 95%)
  σ = Standard deviation of demand
  LeadTime = Supplier lead time (days)
```

### Reorder Point
```
ROP = (Avg Daily Demand × LeadTime) + Safety Stock
```

### Economic Order Quantity
```
EOQ = √(2DS/H)

Where:
  D = Annual demand
  S = Ordering cost per order (~₹100)
  H = Holding cost per unit per year (~25% of cost)
```

### Risk Level
```
Days of Supply = Current Stock / Avg Daily Demand

LOW:       > 100 days
MEDIUM:    30-100 days
HIGH:      10-30 days
CRITICAL:  < 10 days
```

## Tips & Best Practices

### Daily Operations
1. **Record sales within 24 hours**
   - Ensures data accuracy
   - Enables real-time insights

2. **Mark holidays and special events**
   - System learns seasonal patterns
   - Improves forecast accuracy

3. **Track promotional effects**
   - Labels help identify boost patterns
   - Essential for promotion ROI analysis

### Inventory Management
1. **Review inventory weekly**
   - Prevents stockouts
   - Identifies slow-moving items

2. **Update stock accurately**
   - Count physical inventory regularly
   - Match with system records

3. **Follow reorder recommendations**
   - Based on demanb patterns
   - Minimizes carrying costs

### Forecasting
1. **Collect 30+ days data first**
   - Improves prediction accuracy
   - Captures seasonal effects

2. **Update parameters regularly**
   - Festival dates
   - Planned promotions
   - Expected investments

3. **Monitor forecast accuracy**
   - Compare predicted vs. actual
   - Adjust parameters if needed

## Troubleshooting

### Issue: "No Products Found"
**Solution:**
1. View Product Management → View All Products
2. If empty, use Option 2 to add products manually
3. Or use Auto-initialization during registration

### Issue: "Insufficient Data for Prediction"
**Solution:**
1. Need minimum 14 days of data
2. Continue recording daily sales
3. Predictions improve with more data

### Issue: "Store Not Found"
**Solution:**
1. Check store name spelling
2. Ensure conda environment is active
3. Verify data directory exists: `data/user/`

### Issue: Application Crashes
**Solution:**
```bash
# Reactivate environment
conda activate sales_pred

# Run again
python app_enhanced.py

# Check logs for errors
```

## Advanced Features

### Performance Analytics
- Product ranking by revenue
- Category-wise analysis
- Customer demand patterns
- Seasonal trends

### Promotional Analysis
- Promotion effectiveness (% boost)
- Optional vs. regular sales comparison
- ROI calculation potential

### Inventory Optimization
- Avoid overstock (reduce holding costs)
- Prevent stockouts (maintain service level)
- Optimize order quantities

## Future Enhancements

- [ ] Multi-user support
- [ ] Customer name linking
- [ ] Supplier management
- [ ] Margin tracking
- [ ] Automated reordering
- [ ] Mobile app integration
- [ ] Cloud sync
- [ ] Advanced visualizations

## Support & Documentation

- Check main `README.md` for system architecture
- Visit `CONDA_SETUP.md` for environment setup
- See `QUICKSTART.md` for quick start
- Refer to `ARCHITECTURE.md` for technical details

---

**Happy Forecasting! 🚀📊**

For issues or suggestions, please check the project documentation or GitHub repository.
