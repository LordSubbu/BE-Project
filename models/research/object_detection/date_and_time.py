import datetime
now = datetime.datetime.now()
print ("Current date and time : ")
# print (now.strftime("%Y-%m-%d %H:%M:%S"))
x = now.strftime("%d-%m-%y %H:%M %p")
print (x)