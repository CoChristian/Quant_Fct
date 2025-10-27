import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm

# 假设您的 get_mysql_engine 函数类似于以下形式：
def get_mysql_engine(host, user, password, database):
    # 对密码进行URL编码，以防包含特殊字符
    from urllib.parse import quote_plus
    encoded_password = quote_plus(password)

    # 创建连接字符串
    connection_str = f"mysql+pymysql://{user}:{encoded_password}@{host}/{database}"

    # 创建并返回引擎
    engine = create_engine(connection_str)
    return engine

user = 'root'
password = "asdf!@!sadjhAA"

engine = get_mysql_engine(
    host="192.168.0.88",
    user=user,
    password=password,  # 支持特殊字符
    database="quantdb"
)

engine_time_sliced = get_mysql_engine(
    host="192.168.0.88",
    user=user,
    password=password,  # 支持特殊字符
    database="quantdb_time_sliced"
)

try:
    # 编写您的SQL查询语句
    query = "SELECT * FROM stock_price_history "  # 请将 your_table_name 替换为实际表名

    # 使用pandas读取数据:cite[2]:cite[4]
    df = pd.read_sql(query, con=engine)
    print("从quantdb成功读取数据：")
    print(df.head())

except Exception as e:
    print(f"读取数据时出错: {e}")

# 示例2：从 quantdb_time_sliced 数据库读取数据
# read all data
# price_history_data
prefix = 'indicator'
df_vertical = pd.DataFrame()

# 构建一次性查询所有表的SQL
table_names = [f"{prefix}{i:02d}" for i in range(5, 26)]
union_queries = []

for table_name in table_names:
    union_queries.append(f"SELECT *, '{table_name}' as source_table FROM {table_name}")

# 一次性查询所有数据
combined_query = " UNION ALL ".join(union_queries)

try:
    df_vertical = pd.read_sql(combined_query, con=engine_time_sliced)
    print(f"成功读取所有表数据，总行数: {len(df_vertical)}")
except Exception as e:
    print(f"批量读取数据时出错: {e}")





# for i in tqdm(range(5,26)):
#     table_name = prefix + str(0) + str(i)
#     try:
#         # 替换为实际的查询语句
#         another_query = "SELECT * FROM " + table_name  # 请替换为实际查询
#         # query1 = "SELECT * FROM stock_price_history06"
#         # 使用pandas读取数据:cite[2]
#         df_time_sliced = pd.read_sql(another_query, con=engine_time_sliced)
#         # df_time_sliced2 = pd.read_sql(query1, con=engine_time_sliced)
#         print("\n从quantdb_time_sliced成功读取数据：")
#         print(len(df_time_sliced))
#         df_vertical = pd.concat([df_vertical, df_time_sliced], ignore_index=True)
#         # print(df_vertical)
#
#     except Exception as e:
#         print(f"读取时间切片数据时出错: {e}")

#
print(df_vertical)
