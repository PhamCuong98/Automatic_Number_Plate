from flask import Flask, render_template, request
from flaskext.mysql import MySQL
import pickle
import os


app = Flask(__name__)


app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'ngoccuong1812'
app.config['MYSQL_DATABASE_DB'] = 'mydatabase'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'

mysql = MySQL()
mysql.init_app(app)
"""
@app.route('/bienso')
def index():
  cursor = mysql.connect().cursor()
  cursor.execute("SELECT * FROM bienso")
  myresult = cursor.fetchall()
  cursor.close()
  time = []
  bienso=[]
  for i in range(len(myresult)):
    time.append(myresult[i][0])
    bienso.append(myresult[i][1])
  print(".......")
  print(time)
  print(bienso)
  len_n = len(time)
  pics = os.listdir('static/images/')
  return render_template('index.html', time= time, bienso= bienso, len_n=len_n, pics=pics)
"""
database= {'phamcuong':'123', 'hanhnguyen':'123'}
@app.route('/')
def hello_world():
  return render_template("login.html")

@app.route('/accept', methods= ['POST', 'GET'])
def bienso():
  cursor= mysql.connect().cursor()
  cursor.execute("SELECT * FROM bienso")
  myresult = cursor.fetchall()
  cursor.close()
  time = []
  bienso=[]
  for i in range(len(myresult)):
    time.append(myresult[i][0])
    bienso.append(myresult[i][1])
  print(".......")
  print(time)
  print(bienso)
  len_n = len(time)
  return render_template('index.html', time= time, bienso= bienso, len_n=len_n)
def login():
  name= request.form['username']
  pwd= request.form['password']
    
  if name not in database:
    return render_template('login.html', info= "Sai Ten")
  elif database[name] != pwd:
    return render_template('login.html', info= "Sai mat khau")
  else:  
    bienso()

if __name__ == '__main__':
    app.run(debug= True)