import cv2
import time
import requests 
import math
import random 
import numpy as np


# Deteksi objek
model: cv2.dnn.Net = cv2.dnn.readNetFromONNX("Customdata.onnx")
namess = {0: 'Tomat', 1: 'Bawang Merah', 2: 'Bawang Putih', 3: 'Brokoli', 4: 'Kubis', 5: 'Tahu Kuning', 6: 'Tahu Putih', 7: 'Telur', 8: 'Tempe', 9: 'Wortel'}
names = []
for n in namess.keys():
    names.append(namess[n])

cap = cv2.VideoCapture(0)

colors = [
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [255,255,0],
    [0,255,255],
    [255,0,255],
    [255,255,255],
    [0,0,0],
    [128,128,128],
    [128,0,0],    
]
# kirim data ke ubidots
TOKEN = "BBFF-GQ3oWwWR1KPLgldycE0Eu2e4LH1bq8"
DEVICE_LABEL = "Project"
def kirim_data(payload):
    print(payload)
    url = "http://industrial.api.ubidots.com"
    url = "{}/api/v1.6/devices/{}".format(url,DEVICE_LABEL)
    headers = {"X-Auth-Token":TOKEN,"Content-Type":"application/json"}
    status = 400
    attempts = 0
    while status >= 400 and attempts<=5:
        req = requests.post(url=url,headers=headers,json=payload)
        status = req.status_code
        attempts +=1
        time.sleep(1)
    
    print(req.status_code, req.json())
    
    if status>=400:
        print("Error")
        return False
    print("berhasil")
    return True

def build_payload(Jumlah,cara_masak):
    return {"Jumlah":Jumlah,"cara_masak":cara_masak}
# output deteksi
while True:
    _, img = cap.read()
    height, width, _ = img.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = img
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []
    output = outputs[0]
    for i in range(rows):
        classes_scores = output[i][4:]
        minScore, maxScore, minClassLoc, (x, maxClassIndex) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [output[i][0] - 0.5 * output[i][2], output[i][1] - 0.5 * output[i][3],
                   output[i][2], output[i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    
    class_ids_filter = []

    for index in result_boxes:
        box = boxes[index]
        box_out = [round(box[0]*scale), round(box[1]*scale),
                round((box[0] + box[2])*scale), round((box[1] + box[3])*scale)]
        if scores[index] > 0.25:
            cv2.rectangle(img, (box_out[0], box_out[1]), (box_out[2], box_out[3]), colors[class_ids[index]], 2)
            cv2.putText(img, names[class_ids[index]], (box_out[0], box_out[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[class_ids[index]], 2)
            class_ids_filter.append(class_ids[index])       
    # Menghitung Jumlah Objek     
    dict_total_class = {
        'Tomat': 0,
        'Bawang Merah': 0,
        'Bawang Putih': 0,
        'Brokoli': 0,
        'Kubis': 0,
        'Tahu Kuning': 0,
        'Tahu Putih': 0,
        'Telur': 0,
        'Tempe': 0,
        'Wortel': 0
    }
    for i in range(len(class_ids_filter)):
        dict_total_class[names[class_ids_filter[i]]] += 1
    print(dict_total_class)

    # Resep Otomatis
    Resep_1 = """
    1. Bakwan
        Cara memasak :
        - Iris sayuran
        - Cuci sayuran yang telah diiris
        - Masukkan 15 sdm tepung terigu ke dalam wadah yang berisi sayuran
        - Tambahkan kaldu ayam bubuk
        - Aduk semua bahan hingga tercampur merata
        - Panaskan minyak
        - Setelah itu, goreng adonan bakwan hingga kuning kecoklatan
        - Setelah matang, tiriskan bakwan
        - Bakwan siap untuk disantap

    2. Sayur Soup
        Cara memasak :
        - Potong sayuran hingga berukuran kecil,
        - Cuci sayuran tersebut,
        - Masak air hingga mendidih,
        - Setelah air mendidih, masukkan sayuran,
        - Tambahkan kaldu ayam dan pedas bubuk secukupnya dan aduk hingga merata,
        - Tunggu hingga sayur matang,
        - Sayur soup pun siap dihidangkan
    """
    Resep_2 = """
    1. Telur dadar
        Cara memasak :
        - Siapkan 1 butir telur
        - Tuangkan isi telur ke dalam wadah kecil
        - Tambahkan sedikit garam
        - Kocok hingga merata
        - Panaskan minyak
        - Masukkan telur tersebut ke dalam wajan yang berisi minyak panas
        - Tunggu hingga telur berwarna kuning kecoklatan, jangan lupa dibolak balik telurnya
        - Telur siap dimakan
    """
    Resep_3 ="""
    1. Brokoli Lada Hitam.
        Cara memasak : 
        - Iris bawang putih dan potong dadu tomat
        - Tumis bawang hingga harum.
        - Tuang sedikit air. Masukkan cabai dan brokoli.
        - Masukkan bahan saus dan lada hitam, gula, garam, merica, dan kaldu bubuk.
        - Masukkan potongan tomat dan Tuang tepung maizena yang sudah dilarutkan air.
        - Aduk hingga merata dan tunggu hingga matang
        - masakan siap disajikan
    """
    Resep_4 ="""
    1. Brokoli Saus Tiram.
        Cara memasak :
        - Tumis bawang putih, bawang merah hingga muncul aroma wangi.
        - Tuang sedikit air, beri lada hitam dan saus tiram.
        - Beri garam, masukkan brokoli lalu aduk rata.
        - Masukkan tahu dan  masak sampai matang.
        - Brokoli pun siap dihidangkan.
    """
    Resep_5 ="""
    1. Rolade Tahu
        Cara memasak :
        - Haluskan 1 siung bawang merah dan bawang putih, ¼ sdt merica bubuk, ¼ sdt garam
        - Haluskan tahu putih
        - Panaskan sedikit minyak
        - Tumis bumbu yang sudah dihaluskan
        - Masukkan tahu yang sudah dihaluskan
        - Tambahkan garam dan kaldu ayam secukupnya, aduk hingga merata
        - Panaskan kukusan dan masukkan adonan tahu dalam cetakan. Atau buat memanjang seperti lontong, dan bungkus dengan plastik atau daun pisang.
        - Kukuk selama 15 menit. Angkat dan dinginkan.
        - Dadar telur dan bungkus rolade dengan telur.
        - Bisa langsung disajikan atau digoreng lebih dulu.
    """
    Resep_6 ="""
    1. Perkedel Tahu
        Cara memasak : 
        - Haluskan 5 siung bawang putih dan 5 buah cabai 
        - Hancurkan tahu
        - Campur semua bahan dan tambahkan tepung terigu serta kaldu ayam secukupnya
        - Aduk sampai tercampur rata
        - Bentuk adonan perkedel berbentuk bulat. Buat dengan bantuan sendok. Ulangi sampai adonan habis.
        - Panaskan minyak goreng dan masukkan adonan. Cukup balik sekali saja saat warnanya kekuningan. 
        - Angkat saat sudah kering di bagian luar dan  tiriskan.
        - Perkedel tahu siap dimakan.

    2. Tahu Saus Tiram
        Cara memasak : 
        - Potong tahu kecil-kecil, balur dengan tepung serba guna.
        - Panaskan minyak goreng
        - Goreng tahu sampai matang dan agak kering, Angkat lalu tiriskan.
        - Tumis  cabai dan bawang bombai sampai harum.
        - Kemudian masukkan tahu goreng dan saus tiram.
        - Aduk sampai rata. Tambahkan air dan matikan api saat air sudah mengental.
    """
    Resep_7 ="""
    1. Tahu Goreng
        Cara memasak :
        - Panaskan minyak di dalam wajan 
        - Masukkan tahu
        - Tunggu hingga tahu berwarna kuning kecoklatan
        - Tiriskan tahu
        - Tahu siap dimakan
    """
    Resep_8 ="""
    1. Sambal Tempe
        Cara memasak:
        - Potong tempe menjadi berukuran kecil
        - Haluskan 4 siung bawang putih
        - Iris cabe keriting
        - Tumis bumbu halus hingga harum, kemudian tambahkan sedikit air.
        - Tambahkan garam, kaldu bubuk, asem jawa, dan gula merah
        - Aduk hingga merata dan tunggu hingga mengental.
        - Setelah mengental, masukkan tempe, dan cabe kriting, aduk hingga tercampur rata dan kering.
    """
    Resep_9 ="""
    1. Telur Manis Pedas
        Cara memasak : 
        - Goreng telur berbentuk mata sapi
        - Jika sudah matang tiriskan telur
        - Iris 3 siung bawang putih dan bawang merah
        - Potong dadu tomat dan iris cabai
        - Panaskan 2 sdm minyak
        - Tumis bawang dan cabai hingga harum
        - Masukkan sedikit air, garam, penyedap rasa, dan irisan tomat. Aduk sampai mengental.
        - Masukkan telur aduk rata. Angkat dan sajikan.
    """
    Resep_10 ="""
    1. Telur Gulung Saus Asam Manis
        Cara memasak : 
        Membuat telur : 
        - Kocok telur
        - Panaskan sedikit minyak ke dalam wajan
        - Goreng telur tersebut 
        - Jika telur sudah matang, gulung telur tersebut
        - Pindahkan telur ke dalam piring 
        - Potong telur tersebut menjadi beberapa bagian 
        Membuat saus : 
        - Iris bawang putih dan bawang bombay (option)
        - Panaskan sedikit minyak
        - Tumis bawang hingga harum
        - Masukkan saus pedas, saus tomat, dan sedikit garam, aduk hingga merata
        - Tunggu hingga matang
        - Siramkan saus tersebut di atas telur gulung yang sudah dipotong 
        - Telur pun siap dihidangkan 
    """
    Resep_11 ="""
    1. Brokoli Crispy
        Cara memasak :
        - Potong brokoli menjadi beberapa bagian 
        - Cuci brokoli dan rendam menggunakan air garam
        - Masukkan tepung terigu dan tepung tapioka ke dalam wadah, serta tambahkan kaldu ayam bubuk
        - aduk hingga bahan tercampur merata
        - Buang air brokoli dan masukkan brokoli ke dalam adonan yang sudah dibuat, pastikan brokoli tertutupi adonan
        - Panaskan minyak
        - Goreng hingga brokoli berwarna kuning kecoklatan
        - Jika sudah matang, tiriskan brokoli
        - Brokoli siap untuk dihidangkan
    """
    Resep_12 ="""
    1. Tumis Tahu
        Cara memasak :
        - Potong iris bawang putih
        - Ulek tahu hingga hancur
        - Siapkan wajan dan tambahkan 2 sdm minyak
        - Tumis bawang putih hingga harum
        - Masukkan tahu dan tambahkan sedikit garam dan kaldu ayam
        - aduk hingga semua bahan tercampur merata
        - Tunggu hingga tahu matang
        - Tahu siap dihidangkan
    """
    Resep_13 ="""
    1. Sayur Tahu
        Cara masak :
        - Potong tahu berbentuk dadu
        - Iris bawang merah dan bawang putih
        - Potong tomat berbentuk dadu kecil
        - Siapkan wajan dan tambahkan 2 sdm minyak goreng
        - Tumis bawang dan tomat hingga harum
        - Masukkan tahu
        - Tambahkan kaldu ayam dan sedikit gula, aduk hingga merata
        - tambahkan 65 ml santan (sesuai selera) dan aduk kembali
        - Tunggu hingga matang
        - Dan sayur pun siap dihidangkan
    """
    resep = []
    cara_masak = []
    if dict_total_class["Wortel"] >= 1 and dict_total_class["Kubis"] >=1 and dict_total_class["Bawang Merah"] >=1 and dict_total_class["Bawang Putih"] >=1:
        resep.append(" ")
        cara_masak.append(Resep_1)
        dict_total_class["Wortel"]-=1
        dict_total_class["Kubis"] -=1
        dict_total_class["Bawang Merah"]-=1
        dict_total_class["Bawang Putih"] -=1

    if dict_total_class["Telur"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_2)
        dict_total_class["Telur"]-=1
    
    if dict_total_class["Brokoli"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_3)
        dict_total_class["Brokoli"]-=1
        dict_total_class["Tomat"]-=1

    if dict_total_class["Brokoli"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_4)
        dict_total_class["Brokoli"]-=1
        dict_total_class["Tahu Putih"]-=1
        dict_total_class["Bawang Merah"]-=1
        dict_total_class["Bawang putih"]-=1

    if dict_total_class["Tahu Putih"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_5)
        dict_total_class["Tahu Putih"]-=1
        dict_total_class["Brokoli"]-=1
        dict_total_class["Telur"]-=1
        dict_total_class["Bawang Merah"]-=1
        dict_total_class["Bawang Putih"]-=1

    if dict_total_class["Tahu Putih"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_6)
        dict_total_class["Tahu Putih"]-=1
        dict_total_class["Bawang Putih"]-=1

    if dict_total_class["Tahu Kuning"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_7)
        dict_total_class["Tahu Kuning"]-=1

    if dict_total_class["Tempe"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_8)
        dict_total_class["Tempe"]-=1

    if dict_total_class["Telur"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_9)
        dict_total_class["Telur"]-=1
        dict_total_class["Bawang Putih"]-=1
        dict_total_class["Bawang Merah"]-=1

    if dict_total_class["Telur"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_10)
        dict_total_class["Telur"]-=1
        dict_total_class["Bawang Putih"]-=1

    if dict_total_class["Brokoli"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_11)
        dict_total_class["Brokoli"]-=1

    if dict_total_class["Tahu Kuning"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_12)
        dict_total_class["Tahu Kuning"]-=1
        dict_total_class["Bawang Putih"]-=1

    if dict_total_class["Tahu Kuning"] >= 1:
        resep.append(" ")
        cara_masak.append(Resep_13)
        dict_total_class["Tahu Kuning"]-=1
        dict_total_class["Bawang Merah"]-=1
        dict_total_class["Bawang Putih"]-=1
        dict_total_class["Tomat"]-=1

    for i in range(len(resep)):
        print ("Resep : ")
        print(cara_masak[i])
        payload = build_payload(dict_total_class,cara_masak)
        kirim_data(payload)

    cv2.imshow("img", img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()