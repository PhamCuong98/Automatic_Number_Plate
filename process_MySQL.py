import mysql.connector

host= "localhost"
user= "root"
password= "ngoccuong1812"
database= "mydatabase"

class mySQL(object):
    def __init__(self, infor):
        self.infor= infor

    def public(self):
        self.mydb= mysql.connector.connect(host=host, user=user, password=password, database= database)
        self.mycursor= self.mydb.cursor()
        sql= "INSERT INTO bienso (time, day, soxe) VALUES (%s, %s, %s)"
        self.mycursor.execute(sql, self.infor)
        self.mydb.commit()
        print(self.mycursor.rowcount, "Da Insert")

if __name__ == '__main__':
    main()