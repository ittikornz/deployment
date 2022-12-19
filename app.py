# Include PIL, load_image before main()
from pyrsistent import s
import streamlit as st
import os
from PIL import Image
from fastai.vision.all import (
    load_learner,
    PILImage,
)
import pathlib
import pathlib
import streamlit.components.v1 as cpn
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

MAIN_MODEL = pathlib.Path("50ep_resnet50_fastai.pkl")

learn_inf = load_learner(MAIN_MODEL)


def load_image(image_file):
    img = Image.open(image_file)
    return img


st.set_page_config(
    page_title=" Skin Diseases Classification",
    layout="centered",

)


st.title("Skin Diseases Classification")
st.subheader("Something's wrong on your Skin? Let's find it.")

image_file = st.file_uploader("Upload Skin  Images",
                                  type=["png", "jpg", "jpeg","webp"])


def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

if image_file is not None:

    st.image(load_image(image_file), width=None)

    # Saving upload
    with open(os.path.join("./images/upload/", image_file.name), "wb") as f:
        f.write((image_file).getbuffer())


        file = f"./images/upload/{image_file.name}"

        #checkleaf = good_or_bad.predict(file)
        result = learn_inf.predict(file)
        predict = [
            f"<div id='predict' style='background:#FF6B6B;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#fff'>{result[0]}</h1></div>",
            f"<div id='predict' style='background:#6BCB77;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000'>{result[0]}</h1></div>",
            f"<div id='predict' style='background:#FF6B6B;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#000'>{result[0]}</h1></div>",
            f"<div id='predict' style='background:#FFD93D;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#fff'>{result[0]}</h1></div>",
            f"<div id='predict' style='background:#FF6B6B;padding:0 20px 0 20px;border-radius:10px 10px 0 0;'><h1 style='color:#fff'>{result[0]}</h1></div>",
        ]

        diseases = [
                "Atopic Dermatitis Photos",
                "Bullous Disease Photos",
                "Herpes HPV and other STDs Photos",
                "Lupus and other Connective Tissue diseases",
                "Melanoma Skin Cancer Nevi and Moles",
            ]

        info = [
                "To diagnose atopic dermatitis, your health care provider will likely talk with you about your symptoms, examine your skin and review your medical history. You may need tests to identify allergies and rule out other skin diseases.If you think a certain food caused your child's rash, ask your health care provider about potential food allergies.<br/><br/><b>Treatment :</b>Treatment of atopic dermatitis may start with regular moisturizing and other self-care habits. If these don't help, your health care provider might suggest medicated creams that control itching and help repair skin. These are sometimes combined with other treatments.<br/> </div>",
                "Rates of squamous cell and basal cell carcinomas have been increasing, possibly as a result of increased exposure to ultraviolet radiation. Primary care physicians can expect to diagnose six to seven cases of basal cell carcinoma and one to two cases of squamous cell carcinoma each year. Basal cell carcinomas may be plaque-like or nodular with a waxy, translucent appearance, often with ulceration and telangiectasia. They rarely metastasize and are treated with excision, cryotherapy, electrodesiccation and cautery, imiquimod, 5-fluorouracil, or photodynamic therapy (the latter is not approved for this purpose by the U.S. Food and Drug Administration), although surgery results in the fewest recurrences. Actinic keratoses are scaly keratotic patches that often are more easily felt than seen. They are amenable to any of the destructive techniques described above, with the exception of photodynamic therapy. Squamous cell carcinomas arise from keratotic patches and become more nodular and erythematous with growth, sometimes including keratin plugs, horns, or ulceration. Because they may metastasize, they often are treated with excisional biopsy.<br/><br/><b>Treatment :</b><br/> Once a lesion has been identified as suspicious, the question becomes how best to treat it. The method of treatment depends somewhat on the diagnosis, lesion size, morphology, and location, as well as patient compliance. If the pathology shows melanoma, reexcision is required, and lymph node sampling might be considered. If the pathology reveals a basal cell carcinoma or squamous cell carcinoma by excisional biopsy and the margins are clear of malignancy, treatment of that lesion is complete, but the physician and patient should be aware of the risk for additional synchronous or future lesions.</div>",
                "Testing for herpes is not usually part of a routine sexual health scan. If a person has symptoms, they can request a test from their doctor or a sexual health clinic.It is not possible to test for every type of HPV. Also, the infection is so common that an HPV test is not part of a routine sexual health screening. A cervical screening checks for high-risk forms of HPV that can cause cervical cancer.Some people with herpes or HPV have no symptoms or symptoms so mild that a person may not be aware of them.A medical professional will usually only be able to diagnose HPV or herpes if a person has symptoms. After checking the symptoms, they may offer testing if it is available.<br/><br/><b>Treatment :</b><br/>if a healthcare professional detects precancerous changes after testing, a person will undergo further monitoring testsa medical professional can remove a person’s warts, though they rarely cause symptoms and removal is not always necessaryHerpes treatment:a person can take medication to shorten outbreaks reduce their frequencywearing loose clothing can help reduce irritation from blisterskeeping skin clean and dry supports the healing of sorespain relief medication can help with symptoms A person may not require treatment if they have no symptoms or their symptoms are mild.</div>",
                "when typical bullae are present, the diagnosis is suspected clinically. In most cases, the diagnosis will be confirmed by a skin biopsy of an early blister. The diagnosis can also be made from non-blistered, inflamed skin.Pathological examination of bullous pemphigoid shows a split under the epidermis. A dermal neutrophilic infiltrate is usual but not always present. Eosinophils may be prominent.Direct immunofluorescence staining of a skin biopsy taken adjacent to a blister highlights antibodies along the basement membrane that lies between the epidermis and dermis.Blood tests include an indirect immunofluorescence test for circulating pemphigoid BP180 antibodies.<br/><br/><b>Treatment :</b>Ultrapotent topical steroids to treat limited disease < 10% of body surface (eg, clobetasol propionate cream)-Moderate potency topical steroids and emollients to relieve itch and dryness-Systemic steroids (eg, prednisone)-Tetracycline antibiotics, usually doxycycline 200 mg/day; doxycycline has fewer adverse effects than oral corticosteroids and is effective on its own for mild diseas-Other steroid-sparing medications on their own or in combination with steroids-Antibiotics for secondary bacterial infection-Pain relief.<br/> </div>",
                "If you have a mole or other spot that looks suspicious, your doctor may remove it and look at it under the microscope to see if it contains cancer cells. This is called a biopsy.After your doctor receives the skin biopsy results showing evidence of melanoma cells, the next step is to determine if the melanoma has spread. This is called staging. Once diagnosed, melanoma will be categorized based on several factors, such as how deeply it has spread and its appearance under the microscope. Tumor thickness is the most important characteristic in predicting outcomes.'<br/><br/><b>Treatment :</b>Melanoma Surgery: In the early stages, surgery has a high probability of being able to cure your melanoma. Usually performed in an office, a dermatologist numbs the skin with a local anesthetic and removes the melanoma and margins (healthy surrounding skin).Lymphadenectomy: In cases where melanoma has spread, removal of the lymph nodes near the primary diagnosis site may be required. This can prevent the spread to other areas of your body.Metastasectomy: Metastasectomy is used to remove small melanoma bits from organs.Targeted cancer therapy: In this treatment option, drugs are used to attack specific cancer cells. This “targeted” approach goes after cancer cells, leaving healthy cells untouched.Radiation Therapy: Radiation therapy includes treatments with high-energy rays to attack cancer cells and shrink tumors.Immunotherapy: immunotherapy stimulates your own immune system to help fight the cancer.<br/> </div>",
        ]

        index = diseases.index(result[0])

        description = [
            f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #6BCB77'><div><b>ความรุนแรง</b> : <mark style='background-color: #6BCB77;color: white;padding: 10px 10px; border-radius: 40px;font-size:20px'>ปลอดภัย</mark></div><br><b>Diagnosis :</b><br/>{info[index]}",
            f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FFD93D'><div><b>ความรุนแรง</b> : <mark style='background-color: #FFD93D;padding: 10px 10px; border-radius: 40px;font-size:20px'>เฝ้าระวัง</mark></div><br><b>Diagnosis :</b><br/>{info[index]}",
            f"<div style='background:none;padding:30px;border-radius:0 0 10px 10px;border:2px solid #FF6B6B'><div><b>ความรุนแรง</b> : <mark style='background-color: #FF6B6B;color: white;padding: 10px 10px; border-radius: 40px;font-size:20px'>อันตราย</mark></div><br><b>Diagnosis :</b><br/>{info[index]}",
        ]
        st.markdown(predict[index], unsafe_allow_html=True)
        if(index == 1):
            st.markdown(description[0], unsafe_allow_html=True)
        elif(index == 3):
            st.markdown(description[1], unsafe_allow_html=True)
        elif(index == 0 or index == 2 or index == 4):
            st.markdown(description[2], unsafe_allow_html=True)


