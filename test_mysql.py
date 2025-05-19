import mysql.connector
from mysql.connector import Error

try:
    connection = mysql.connector.connect(
        host='localhost',
        port=3306,
        user='asr_user',
        password='123',
        database='asr_db'
    )

    if connection.is_connected():
        print("✅ Connexion réussie à la base de données MySQL !")

except Error as e:
    print(f"❌ Erreur de connexion : {e}")

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("🔌 Connexion fermée.")
