from django.http import HttpResponse
from django.shortcuts import render
from .models import *
from django.shortcuts import redirect
# FILE UPLOAD AND VIEW
from  django.core.files.storage import FileSystemStorage
# SESSION
from django.conf import settings
import os
from ML import test
from ML import test_bone
from django.utils import timezone



# def first(request):
#     return render(request,'index.html')

def index(request):
    return render(request,'index.html')

def register(request):
    return render(request,'register.html')

def addregister(request):
    if request.method=="POST":
        name=request.POST.get('Name')
        email=request.POST.get('Email')
        password=request.POST.get('Password')
        phone=request.POST.get('Phone Number')

        reg=Register(name=name,email=email,password=password,phone=phone)
        reg.save()
    return render(request,'index.html')

def login(request):
    return render(request,'login.html')

def addlogin(request):
    email = request.POST.get('Email')
    password = request.POST.get('Password')
    if email == 'admin@gmail.com' and password =='admin':
        request.session['adminid'] = email
        request.session['admin'] = 'admin'
        return render(request,'index.html')

    elif Register.objects.filter(email=email,password=password).exists():
        userdetails=Register.objects.get(email=request.POST['Email'], password=password)
        if userdetails.password == request.POST['Password']:
            request.session['uid'] = userdetails.id
        
        return render(request,'index.html')
        
    else:
        return render(request, 'login.html')
    
def logout(request):
    session_keys = list(request.session.keys())
    for key in session_keys:
        del request.session[key]
    return redirect(index)


def view_user(request):
    result = Register.objects.all()
    return render(request,"viewuser.html", {'result':result})



def upload(request):
    return render(request,'upload.html' ,{"disease":["Heart Disease","Alzheimer's Disease","Bone Cancer"]})


def addupload(request):
    user_id = request.session["uid"]
    if request.method=="POST":
        disease = request.POST.get("disease")
        test_file=request.FILES.get("file")
        base_input_folder =settings.MEDIA_ROOT

        if disease == "Heart Disease":

            try:
                os.remove(os.path.join(base_input_folder,"heart","test.csv"))
            except:
                pass

            fs =FileSystemStorage(location = os.path.join(base_input_folder,"heart"))
            fs.save("test.csv",test_file)
            pred_result= test.predict_heart()

        elif disease == "Alzheimer's Disease":
            try:
                os.remove(os.path.join(base_input_folder,"alz","test.jpg"))
            except:
                pass

            fs = FileSystemStorage(location =os.path.join(base_input_folder,"alz"))
            fs.save("test.jpg",test_file)
            pred_result = test.predict_alz()


        else:
            try:
                os.remove(os.path.join(base_input_folder,"bone","test.jpg"))
            except:
                pass

            fs = FileSystemStorage(location =os.path.join(base_input_folder,"bone"))
            fs.save("test.jpg",test_file)
            pred_result = test_bone.process_image()
            

        insert_into_table = Result(user_id = user_id , result = pred_result, datetime=timezone.now())
        insert_into_table.save()
        return render(request, "result.html",{ 'result':pred_result , 'mssg' : "Result"})




def result(request):
    return render(request,"result.html", {'result':"Please upload test file first!!"})


def profile(request):
    profile_id = request.session.get('uid')
    profile = Register.objects.get(id=profile_id)
    print("dddddddddddddddddddddd",profile, profile_id)
    return render(request, 'profile.html', {'profile': profile})

def edit_profile(request):
    user = request.session.get('uid')
    profile = Register.objects.get(id=user)
    if request.method == "POST":
        name = request.POST.get('name')
        phone = request.POST.get('phone')
        email = request.POST.get('email')
        password = request.POST.get('password')

        profile.name = name
        profile.phone = phone
        profile.email = email
        profile.password = password
        profile.save()
        return redirect('profile')

    return render(request, 'edit_profile.html', {'profile': profile})
