import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL' : 'https://faceattendancerealtime-790e0-default-rtdb.firebaseio.com/'
})

ref = db.reference('Students')


data = {
    "125252":
        {
            'name':'Raghu M',
            'major': 'Data Science',
            'starting_year': 2020,
            'total_attendance': 10,
            'standing': 'good',
            'year': 2,
            'last_attendance_time': '2023-10-10 00:54:34'
        },
    "755318":
        {
            'name': 'Elon Musk',
            'major': 'Electric Vehicles',
            'starting_year': 2021,
            'total_attendance': 10,
            'standing': 'bad',
            'year': 2,
            'last_attendance_time': '2023-10-10 00:54:34'
        },
    "944125":
        {
            'name': 'Seb',
            'major': 'Formula 1',
            'starting_year': 2020,
            'total_attendance': 10,
            'standing': 'Very poor',
            'year': 2,
            'last_attendance_time': '2023-10-10 00:54:34'
        },
    "984636":
        {
            'name': 'Charles Leclerc',
            'major': 'Formula 1',
            'starting_year': 2020,
            'total_attendance': 1,
            'standing': 'Very poor',
            'year': 2,
            'last_attendance_time': '2023-10-10 00:54:34'
        },
    "123123":
        {
            'name': 'Yilun Shang',
            'major': 'Statistics',
            'starting_year': 2020,
            'total_attendance': 10,
            'standing': 'good',
            'year': 2,
            'last_attendance_time': '2023-10-10 00:54:34'
        },
    "222333":
        {
            'name': 'Joj',
            'major': 'Full Stack',
            'starting_year': 2020,
            'total_attendance': 10,
            'standing': 'poor',
            'year': 2,
            'last_attendance_time': '2023-10-10 00:54:34'
        },
}

for key, value in data.items():
    ref.child(key).set(value)