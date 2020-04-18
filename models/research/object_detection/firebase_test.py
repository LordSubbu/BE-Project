from firebase import firebase


firebase =  firebase.FirebaseApplication("https://python-test-1235f.firebaseio.com/",None)

data = {
    'Name':'Navin',
    'Email':'ggraosfkjgd'
}

result = firebase.post('/Hellosucker', data)
print(result)
