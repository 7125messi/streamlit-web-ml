import streamlit as st
st.write('''
    * 缓存是键值存储，其中键是以下各项的哈希：
        * 您用来调用函数的输入参数
        * 函数中使用的任何外部变量的值
        * 功能主体
        * 缓存函数内部使用的任何函数的主体
    
    * 该值是一个元组：
        * 缓存的输出
        * 缓存输出的哈希值
    
    * 对于键和输出哈希，Streamlit使用专门的哈希函数，该函数知道如何遍历代码，哈希特殊对象，并且可以由用户自定义。

    For example, when the function expensive_computation(a, b), decorated with @st.cache, is executed with a=2 and b=21, Streamlit does the following:
    * Computes the cache key
    * If the key is found in the cache, then:
        * Extracts the previously-cached (output, output_hash) tuple.
        * Performs an **Output Mutation Check**, where a fresh hash of the output is computed and compared to the stored output_hash.
            * If the two hashes are different, shows a **Cached Object Mutated warning**. (Note: Setting **allow_output_mutation=True** disables this step).
    * If the input key is not found in the cache, then:
        * Executes the cached function (i.e. output = expensive_computation(2, 21)).
        * Calculates the output_hash from the function’s output.
        * Stores key → (output, output_hash) in the cache.
    * Returns the output.

    * hash_funcs参数：允许自定义哈希函数
    ```python
    class FileReference:
        def __init__(self, filename):
            self.filename = filename

    def hash_file_reference(file_reference):
        with open(file_reference.filename) as f:
        return f.read()

    @st.cache(hash_funcs={FileReference: hash_file_reference})
    def func(file_reference):
        pass
    ```

    * python中典型的哈希函数
        * Python的id功能：适合单例对象，例如打开的数据库连接或TensorFlow会话。这些对象将只实例化一次，无论脚本重新运行多少次。


    # 示例1：传递数据库连接

    假设我们要打开一个数据库连接，该连接可以在Streamlit应用程序的多次运行中重用。为此，您可以利用以下事实：通过引用存储缓存的对象以自动初始化和重用连接
    ```python
    @st.cache(allow_output_mutation=True)
    def get_database_connection():
        return db.get_connection()
    ```
    仅用3行代码，数据库连接就创建一次并存储在缓存中。然后，在以后的每一次get_database_conection调用中，已创建的连接对象将自动重用。换句话说，它成为一个单例。


    如果要编写一个函数来接收数据库连接作为输入怎么办？为此，您将使用hash_funcs：
    ```python
    @st.cache(hash_funcs={DBConnection:id})
    def get_users(connection):
        # Note: We assume that connection is of type DBConnection.
        return connection.execute_sql('SELECT * from Users')
    ```

    在这里，我们使用Python的内置id函数，因为连接对象通过该get_database_conection函数来自Streamlit缓存。
    这意味着每次都会传递相同的连接实例，因此它始终具有相同的ID。
    但是，如果您碰巧在第二个连接对象周围指向了一个完全不同的数据库，则将其传递给仍然是安全的，get_users因为可以保证其ID与第一个ID不同。
    只要您拥有指向外部资源的对象（例如数据库连接或Tensorflow会话），这些设计模式就会应用。
''')
import pymysql

@st.cache(allow_output_mutation=True)
def get_database_connection():
    """
    获取MySQL的链接
    :return: mysql connection
    """
    st.write('我被缓存命中了')
    return pymysql.connect(
        host = '192.168.1.2',
        port = 3306,
        user = 'root',
        password = 'ydzhao',
        database = 'python_mysql',
        charset = 'utf8'
    )

@st.cache(hash_funcs={get_database_connection:id})
def query_data(sqlStr):
    df = get_database_connection().cursor().execute(sqlStr)
    return df
         


if __name__ == "__main__":
    sqlStr = 'select * from iris_data'
    query_data(sqlStr)