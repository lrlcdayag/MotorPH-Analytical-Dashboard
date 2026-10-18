# Code summary
The MotorPH dashboard script reads the cleaned product catalog and sales CSVs, validates and prepares the data, then generates two dashboard files: a static PNG and an interactive HTML page.

# 1. Loads and checks the datasets
It reads:
- Cleaned products CSV
- Cleaned Q3 sales CSV
Before building the dashboard, it checks that required columns exist, removes exact duplicate rows, trims text values, converts dates and numeric fields, and removes sales rows that have missing or invalid required values.

It also checks that:
- Product names are unique in the catalog.
- Product prices, sales prices, quantities, and totals are positive whole numbers.
- Each sales product exists in the catalog.
- The sales unit price matches the catalog price.
- Each line total equals unit price × quantity.
- If an important validation fails such as a product not matching the catalog, the script raises an error instead of silently producing a potentially misleading dashboard.
  
# 2. Groups small categories
The script calculates each category’s share of the product catalog. Categories representing less than 5% of products are combined into “Other (<5%)” for the category charts, reducing visual clutter. The inventory table still shows each product’s original category.

# 3. Calculates summary information
The script calculates total revenue and units sold, counts sales rows, products, and categories, and summarizes sales by month, product, and category. Months with no sales records between the first and last sales month are included with zero values in the monthly charts.

It also reports how many sales rows have “Unknown” client-type and payment-method labels. Those values are retained rather than imputed.

# 4. Creates the dashboard visuals
Both versions include:
- Overview metrics: total revenue, units sold, sales rows, product count, and category count.
- Sales performance: monthly revenue and monthly units sold.
- Product performance: the top 10 products by revenue and by units sold.
- Category performance: catalog product counts and sales revenue by category.
- Inventory: every product’s category, unit price, units sold, and revenue.

# 5. Produces two outputs
- PNG dashboard: a static image for reports or submissions.
- Interactive HTML dashboard: opens in a browser and supports chart hover details and Plotly’s zoom, pan, and reset controls. Plotly is embedded in the file, so the dashboard does not need to download its charting library when opened.
