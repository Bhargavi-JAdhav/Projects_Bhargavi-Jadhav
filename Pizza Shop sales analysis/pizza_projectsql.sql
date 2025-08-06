SELECT * FROM mytestdb.pizza_sales;
ALTER TABLE mytestdb.pizza_sales MODIFY order_date DATE;
ALTER TABLE mytestdb.pizza_sales MODIFY order_time time;
-- 1. to get total sales
SELECT 
    sum(total_price) as total_revenue
FROM
    mytestdb.pizza_sales ;
select sum(total_price)/count(distinct order_id) as average_order_value from mytestdb.pizza_sales;
-- total pizza sold
select sum(quantity) as total_pizza from mytestdb.pizza_sales;
-- total orders placed
select max(order_id) as total_order from mytestdb.pizza_sales;
select count(distinct order_id) as total_order from mytestdb.pizza_sales;
select sum(quantity)/count(distinct order_id) as average_order_pizza from mytestdb.pizza_sales;

SELECT DAYNAME(order_date) AS order_day, 
       COUNT(DISTINCT order_id) AS total_orders 
FROM mytestdb.pizza_sales
GROUP BY DAYNAME(order_date);
SELECT order_day, total_orders
FROM (
    SELECT DAYNAME(order_date) AS order_day, 
           COUNT(DISTINCT order_id) AS total_orders 
    FROM mytestdb.pizza_sales
    GROUP BY DAYNAME(order_date)
) AS subquery
ORDER BY FIELD(order_day, 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday');
-- hourly trend
SELECT HOUR(order_time) AS order_hour, 
       COUNT(DISTINCT order_id) AS total_orders 
FROM mytestdb.pizza_sales
GROUP BY HOUR(order_time)
ORDER BY order_hour;
-- based on pizza size
SELECT 
    pizza_size, 
    SUM(total_price) * 100 / (SELECT SUM(total_price) FROM mytestdb.pizza_sales WHERE QUARTER(order_date) = 1) AS percentage_of_total,
    SUM(total_price) AS total_price
FROM 
    mytestdb.pizza_sales
WHERE 
    QUARTER(order_date) = 1
GROUP BY 
    pizza_size;
    -- total pizza sold by pizza category
    select pizza_category, sum(quantity) as pizza_sold_categorywise from mytestdb.pizza_sales 
    group by pizza_category;
    select pizza_name, sum(quantity) as pizza_sold_namewise from mytestdb.pizza_sales 
    group by pizza_name
    order by sum(quantity) asc limit 5;





