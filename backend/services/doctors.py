from typing import Dict, Any, Optional, List

# Doctor mapping: state -> disease -> details
# diseases: "Diabetic Retinopathy", "Glaucoma", "Cataract", "Normal"
DOCTORS_BY_STATE: Dict[str, Dict[str, Dict[str, str]]] = {
    "Andhra Pradesh": {
        "Diabetic Retinopathy": {
            "name": "Dr. K. Srinivas Rao",
            "specialization": "Vitreo-Retinal Surgeon",
            "clinic": "LV Prasad Eye Institute, Vijayawada",
            "contact": "+91 866 230 4966 | lvpei@lvpei.org"
        },
        "Glaucoma": {
            "name": "Dr. Padma Rekha",
            "specialization": "Glaucoma Specialist",
            "clinic": "Vasan Eye Care, Visakhapatnam",
            "contact": "+91 891 255 6677 | vizag@vasaneye.in"
        },
        "Cataract": {
            "name": "Dr. Ravi Teja",
            "specialization": "Phaco Surgeon",
            "clinic": "Sarada Eye Hospital, Guntur",
            "contact": "+91 863 222 1111 | info@saradaeye.com"
        },
        "Normal": {
            "name": "AP Eye Health Services",
            "specialization": "Primary Eye Care",
            "clinic": "Govt Eye Hospital, Vijayawada",
            "contact": "Helpline: 104"
        }
    },
    "Arunachal Pradesh": {
        "Diabetic Retinopathy": {
            "name": "Dr. Tage Ratan",
            "specialization": "Retina Specialist",
            "clinic": "Itanagar Eye Care Centre",
            "contact": "+91 360 221 2345 | info@itanagareye.org"
        },
        "Glaucoma": {
            "name": "Dr. Bengia Wangsu",
            "specialization": "Ophthalmologist",
            "clinic": "TRIHMS Eye Department, Naharlagun",
            "contact": "+91 360 222 4567 | trihms@nic.in"
        },
        "Cataract": {
            "name": "Dr. Millo Taba",
            "specialization": "General Ophthalmologist",
            "clinic": "State Referral Hospital, Itanagar",
            "contact": "+91 360 221 3456 | srh_arunachal@nic.in"
        },
        "Normal": {
            "name": "Arunachal Eye Services",
            "specialization": "Preventive Ophthalmology",
            "clinic": "District Hospital Eye Unit",
            "contact": "Helpline: 104"
        }
    },
    "Assam": {
        "Diabetic Retinopathy": {
            "name": "Dr. Bimal Borah",
            "specialization": "Senior Retina Surgeon",
            "clinic": "Sankara Eye Hospital, Guwahati",
            "contact": "+91 361 254 5678 | guwahati@sankaraeye.com"
        },
        "Glaucoma": {
            "name": "Dr. Priti Choudhury",
            "specialization": "Glaucoma Consultant",
            "clinic": "Chetana Eye Hospital, Guwahati",
            "contact": "+91 361 246 0000 | info@chetanaeye.in"
        },
        "Cataract": {
            "name": "Dr. Anjan Nath",
            "specialization": "Cataract & IOL Surgeon",
            "clinic": "Gauhati Medical College Eye Dept",
            "contact": "+91 361 252 9457 | gmceye@nic.in"
        },
        "Normal": {
            "name": "Assam Eye Health Unit",
            "specialization": "Community Eye Care",
            "clinic": "NRHM Eye Centre, Guwahati",
            "contact": "Helpline: 104"
        }
    },
    "Bihar": {
        "Diabetic Retinopathy": {
            "name": "Dr. Anil Kumar Singh",
            "specialization": "Vitreo-Retinal Specialist",
            "clinic": "Drishti Eye Centre, Patna",
            "contact": "+91 612 222 8888 | dr.anil@drishtipatna.com"
        },
        "Glaucoma": {
            "name": "Dr. Sunita Kumari",
            "specialization": "Glaucoma Specialist",
            "clinic": "IGIMS Eye Department, Patna",
            "contact": "+91 612 229 7531 | igims@nic.in"
        },
        "Cataract": {
            "name": "Dr. Rajesh Prasad",
            "specialization": "Phacoemulsification Surgeon",
            "clinic": "Sarojini Eye Hospital, Patna",
            "contact": "+91 612 234 5678 | sarojini@eye.in"
        },
        "Normal": {
            "name": "Bihar NPCB Eye Services",
            "specialization": "Blindness Control",
            "clinic": "Patna Medical College Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Chhattisgarh": {
        "Diabetic Retinopathy": {
            "name": "Dr. Deepak Tiwari",
            "specialization": "Retina Specialist",
            "clinic": "Shri Shankaracharya Eye Hospital, Raipur",
            "contact": "+91 771 222 5555 | info@ssieye.com"
        },
        "Glaucoma": {
            "name": "Dr. Nirmala Patel",
            "specialization": "Glaucoma Consultant",
            "clinic": "Ramkrishna Care Hospital Eye Dept",
            "contact": "+91 771 407 4444 | eye@raipur.com"
        },
        "Cataract": {
            "name": "Dr. Suresh Verma",
            "specialization": "Cataract Surgeon",
            "clinic": "Pt JNM Medical College Eye Dept",
            "contact": "+91 771 224 1234 | jnm@nic.in"
        },
        "Normal": {
            "name": "CG Eye Services",
            "specialization": "Primary Eye Care",
            "clinic": "District Hospital Eye Unit, Raipur",
            "contact": "Helpline: 104"
        }
    },
    "Delhi": {
        "Diabetic Retinopathy": {
            "name": "Dr. Sameer Gupta",
            "specialization": "Senior Vitreo-Retinal Surgeon",
            "clinic": "Delhi Eye Care Centre",
            "contact": "+91 98123 45678 | dr.sameer@delhieye.com"
        },
        "Glaucoma": {
            "name": "Dr. Anita Sharma",
            "specialization": "Glaucoma Specialist",
            "clinic": "Vision Hospital, New Delhi",
            "contact": "+91 98765 43210 | anita.s@vision.in"
        },
        "Cataract": {
            "name": "Dr. Karan Mehra",
            "specialization": "Phaco & Refractive Surgeon",
            "clinic": "Sight Institute, Delhi",
            "contact": "+91 95555 44433 | contact@sightdelhi.com"
        },
        "Normal": {
            "name": "General OPD Delhi",
            "specialization": "Comprehensive Eye Care",
            "clinic": "AIIMS Eye Center",
            "contact": "Local Help-line: 104"
        }
    },
    "Goa": {
        "Diabetic Retinopathy": {
            "name": "Dr. Rohan Naik",
            "specialization": "Vitreoretinal Specialist",
            "clinic": "Goa Medical College Eye Dept, Panaji",
            "contact": "+91 832 245 8000 | gmceye@goa.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Anushka Dessai",
            "specialization": "Glaucoma Consultant",
            "clinic": "Eye Q Vision, Panaji",
            "contact": "+91 832 242 2222 | panaji@eyeqvision.in"
        },
        "Cataract": {
            "name": "Dr. Savio Fernandes",
            "specialization": "Cataract & IOL Surgeon",
            "clinic": "Bhaktivedanta Eye Hospital, Margao",
            "contact": "+91 832 271 4444 | margao@bveye.com"
        },
        "Normal": {
            "name": "Goa Health Eye Wing",
            "specialization": "Primary Ophthalmology",
            "clinic": "Goa Medical College OPD",
            "contact": "Helpline: 104"
        }
    },
    "Gujarat": {
        "Diabetic Retinopathy": {
            "name": "Dr. Chirag Shah",
            "specialization": "Senior Retina Surgeon",
            "clinic": "Amardeep Eye Hospital, Ahmedabad",
            "contact": "+91 79 2640 1000 | info@amardeep.com"
        },
        "Glaucoma": {
            "name": "Dr. Hetal Patel",
            "specialization": "Glaucoma Specialist",
            "clinic": "Narayana Eye Hospital, Surat",
            "contact": "+91 261 222 5555 | surat@narayana.in"
        },
        "Cataract": {
            "name": "Dr. Amit Vora",
            "specialization": "Microincision Surgeon",
            "clinic": "Netradeep Eye Care, Rajkot",
            "contact": "+91 281 245 4545 | rajkot@netradeep.in"
        },
        "Normal": {
            "name": "Gujarat Health Care",
            "specialization": "Blindness Prevention Unit",
            "clinic": "M&J Regional Institute, Ahmedabad",
            "contact": "Toll Free: 1800 233 4455"
        }
    },
    "Haryana": {
        "Diabetic Retinopathy": {
            "name": "Dr. Sunil Arora",
            "specialization": "Retina & Uvea Specialist",
            "clinic": "Pushpanjali Eye Centre, Gurgaon",
            "contact": "+91 124 428 9000 | info@pushpanjalieye.com"
        },
        "Glaucoma": {
            "name": "Dr. Kavita Rani",
            "specialization": "Glaucoma HOD",
            "clinic": "PGIMS Eye Department, Rohtak",
            "contact": "+91 1262 211 307 | pgims@hry.gov.in"
        },
        "Cataract": {
            "name": "Dr. Vikrant Nain",
            "specialization": "Phaco Surgeon",
            "clinic": "Eye Mantra, Faridabad",
            "contact": "+91 124 414 2222 | faridabad@eyemantra.in"
        },
        "Normal": {
            "name": "Haryana Eye Health",
            "specialization": "Primary Eye Care",
            "clinic": "Civil Hospital Eye OPD, Ambala",
            "contact": "Helpline: 104"
        }
    },
    "Himachal Pradesh": {
        "Diabetic Retinopathy": {
            "name": "Dr. Ranjit Kanwar",
            "specialization": "Retina Specialist",
            "clinic": "IGMC Eye Department, Shimla",
            "contact": "+91 177 280 4251 | igmc@hp.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Poonam Sharma",
            "specialization": "Glaucoma Surgeon",
            "clinic": "Dr. RPGMC Eye Dept, Tanda (Kangra)",
            "contact": "+91 1892 267 114 | rpgmc@hp.gov.in"
        },
        "Cataract": {
            "name": "Dr. Anil Sood",
            "specialization": "Cataract Surgeon",
            "clinic": "Drishti Eye Centre, Sundernagar",
            "contact": "+91 1907 266 789 | drishti@hp.in"
        },
        "Normal": {
            "name": "HP Eye Health",
            "specialization": "Community Eye Care",
            "clinic": "Zonal Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Jharkhand": {
        "Diabetic Retinopathy": {
            "name": "Dr. Manish Kumar",
            "specialization": "Vitreoretinal Surgeon",
            "clinic": "Vivekananda Eye Hospital, Ranchi",
            "contact": "+91 651 246 7890 | info@viveye.in"
        },
        "Glaucoma": {
            "name": "Dr. Priya Sinha",
            "specialization": "Glaucoma Specialist",
            "clinic": "Raj Hospital Eye Dept, Ranchi",
            "contact": "+91 651 234 5679 | rajhospital@jhar.in"
        },
        "Cataract": {
            "name": "Dr. Santosh Gupta",
            "specialization": "IOL & Phaco Surgeon",
            "clinic": "RIMS Eye Department, Ranchi",
            "contact": "+91 651 245 5660 | rims@jhar.gov.in"
        },
        "Normal": {
            "name": "Jharkhand Eye Services",
            "specialization": "Primary Eye Care",
            "clinic": "Sadar Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Karnataka": {
        "Diabetic Retinopathy": {
            "name": "Dr. Rajesh Hegde",
            "specialization": "Vitreoretinal Specialist",
            "clinic": "Bangalore Retina Institute",
            "contact": "+91 80 4123 4567 | care@blr-retina.in"
        },
        "Glaucoma": {
            "name": "Dr. Meera Rao",
            "specialization": "Glaucoma & Anterior Segment",
            "clinic": "Narayana Nethralaya",
            "contact": "+91 80 6612 1212 | dr.meera@narayana.com"
        },
        "Cataract": {
            "name": "Dr. Santosh Kumar",
            "specialization": "Cataract Surgeon",
            "clinic": "Manipal Eye Care, Bangalore",
            "contact": "+91 80 2502 4444 | santosh.k@manipal.edu"
        },
        "Normal": {
            "name": "Vision Care Karnataka",
            "specialization": "General Ophthalmology",
            "clinic": "Govt Eye Hospital, Bangalore",
            "contact": "Helpline: 080-12345"
        }
    },
    "Kerala": {
        "Diabetic Retinopathy": {
            "name": "Dr. Jayakrishna Nair",
            "specialization": "Vitreo-Retinal Surgeon",
            "clinic": "Chaithanya Eye Hospital, Thiruvananthapuram",
            "contact": "+91 471 272 4400 | info@chaithanyaeye.com"
        },
        "Glaucoma": {
            "name": "Dr. Suma Raveendran",
            "specialization": "Glaucoma HOD",
            "clinic": "Amrita Eye Care, Kochi",
            "contact": "+91 484 230 1234 | eyecare@amrita.edu"
        },
        "Cataract": {
            "name": "Dr. Thomas Mathew",
            "specialization": "Phaco & Refractive Surgeon",
            "clinic": "Little Flower Hospital Eye Dept, Angamaly",
            "contact": "+91 484 245 5012 | lfh@angamaly.in"
        },
        "Normal": {
            "name": "Kerala Eye Health",
            "specialization": "Primary Ophthalmology",
            "clinic": "Govt Medical College Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Madhya Pradesh": {
        "Diabetic Retinopathy": {
            "name": "Dr. Sandeep Mehta",
            "specialization": "Retina Specialist",
            "clinic": "Netralaya Eye Hospital, Bhopal",
            "contact": "+91 755 246 8950 | info@netralayabhopal.com"
        },
        "Glaucoma": {
            "name": "Dr. Ritu Jain",
            "specialization": "Glaucoma Specialist",
            "clinic": "Gandhi Medical College Eye Dept",
            "contact": "+91 755 254 0222 | gmc@mp.gov.in"
        },
        "Cataract": {
            "name": "Dr. Kamlesh Shrivastava",
            "specialization": "Cataract Surgeon",
            "clinic": "Shri Aurobindo Institute Eye Dept",
            "contact": "+91 731 245 0000 | saims@mp.in"
        },
        "Normal": {
            "name": "MP NPCB Eye Unit",
            "specialization": "Community Eye Care",
            "clinic": "Hamidia Hospital Eye OPD, Bhopal",
            "contact": "Helpline: 104"
        }
    },
    "Maharashtra": {
        "Diabetic Retinopathy": {
            "name": "Dr. Prerna Patil",
            "specialization": "Retina Expert",
            "clinic": "Mumbai Eye Foundation",
            "contact": "+91 22 4000 5000 | info@mumbaieye.org"
        },
        "Glaucoma": {
            "name": "Dr. Vikram Deshmukh",
            "specialization": "HOD Glaucoma Dept",
            "clinic": "Pune Retinal Hospital",
            "contact": "+91 20 2567 8900 | vikram.d@puneretina.com"
        },
        "Cataract": {
            "name": "Dr. Sneha Kulkarni",
            "specialization": "Senior Consultant",
            "clinic": "Nagpur Eye Clinic",
            "contact": "+91 712 222 3333 | drsneha@nagpureye.com"
        },
        "Normal": {
            "name": "Ophthalmic Care MH",
            "specialization": "Preventive Eye Care",
            "clinic": "State Govt Ophthalmic Unit",
            "contact": "Toll Free: 1800 123 456"
        }
    },
    "Manipur": {
        "Diabetic Retinopathy": {
            "name": "Dr. Thangjam Iboto Singh",
            "specialization": "Retina Specialist",
            "clinic": "RIMS Eye Department, Imphal",
            "contact": "+91 385 245 1892 | rims@manipur.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Laishram Sonia Devi",
            "specialization": "Ophthalmologist",
            "clinic": "Shija Hospitals Eye Unit, Imphal",
            "contact": "+91 385 244 1234 | eye@shijahospitals.com"
        },
        "Cataract": {
            "name": "Dr. Haobam Suresh",
            "specialization": "IOL Surgeon",
            "clinic": "Imphal Eye Care Centre",
            "contact": "+91 385 220 5678 | imphalec@mn.in"
        },
        "Normal": {
            "name": "Manipur Eye Health",
            "specialization": "Primary Eye Care",
            "clinic": "District Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Meghalaya": {
        "Diabetic Retinopathy": {
            "name": "Dr. Donboklang Mawlong",
            "specialization": "Retina Specialist",
            "clinic": "NEIGRIHMS Eye Dept, Shillong",
            "contact": "+91 364 253 8000 | neigrihms@nic.in"
        },
        "Glaucoma": {
            "name": "Dr. Pynshailang Siangshai",
            "specialization": "Glaucoma Consultant",
            "clinic": "Nazareth Hospital Eye Dept, Shillong",
            "contact": "+91 364 250 1248 | nazareth@meg.in"
        },
        "Cataract": {
            "name": "Dr. Beniameen Marak",
            "specialization": "General Ophthalmologist",
            "clinic": "Civil Hospital Eye Unit, Shillong",
            "contact": "+91 364 222 3456 | civilhospital@meg.gov.in"
        },
        "Normal": {
            "name": "Meghalaya Eye Services",
            "specialization": "Community Eye Health",
            "clinic": "CH OPD Eye Unit",
            "contact": "Helpline: 104"
        }
    },
    "Mizoram": {
        "Diabetic Retinopathy": {
            "name": "Dr. Lalmalsawma Hmar",
            "specialization": "Retina Consultant",
            "clinic": "Civil Hospital Eye Dept, Aizawl",
            "contact": "+91 389 323 2177 | civilhospital@miz.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Vanlalhriatpuii",
            "specialization": "Ophthalmologist",
            "clinic": "Synod Hospital Eye Care, Aizawl",
            "contact": "+91 389 232 3456 | synod@aizawl.in"
        },
        "Cataract": {
            "name": "Dr. Zorammawia Thanga",
            "specialization": "IOL Surgeon",
            "clinic": "Dawrpui Eye Clinic, Aizawl",
            "contact": "+91 389 232 5678 | dawrpui@miz.in"
        },
        "Normal": {
            "name": "Mizoram Eye Health",
            "specialization": "Primary Eye Care",
            "clinic": "District Hospital OPD",
            "contact": "Helpline: 104"
        }
    },
    "Nagaland": {
        "Diabetic Retinopathy": {
            "name": "Dr. Kevichusa Medikhru",
            "specialization": "Retina Specialist",
            "clinic": "Naga Hospital Authority, Kohima",
            "contact": "+91 370 229 0123 | nha@nagaland.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Vikuonuo Merry",
            "specialization": "Glaucoma Consultant",
            "clinic": "Christian Institute Eye Dept, Dimapur",
            "contact": "+91 386 224 5678 | cie@dimapur.in"
        },
        "Cataract": {
            "name": "Dr. Seyieselie Sekhose",
            "specialization": "Cataract Surgeon",
            "clinic": "NHAK Eye Unit, Kohima",
            "contact": "+91 370 229 4567 | nhak@nag.gov.in"
        },
        "Normal": {
            "name": "Nagaland Eye Services",
            "specialization": "Community Ophthalmology",
            "clinic": "District Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Odisha": {
        "Diabetic Retinopathy": {
            "name": "Dr. Suresh Panda",
            "specialization": "Vitreoretinal Surgeon",
            "clinic": "L V Prasad Eye Institute, Bhubaneswar",
            "contact": "+91 674 398 2903 | bbs@lvpei.org"
        },
        "Glaucoma": {
            "name": "Dr. Mamata Mishra",
            "specialization": "Glaucoma Specialist",
            "clinic": "Hi-Tech Medical College Eye Dept",
            "contact": "+91 674 235 6779 | hitech@odi.in"
        },
        "Cataract": {
            "name": "Dr. Biswabhusan Routray",
            "specialization": "Phaco Surgeon",
            "clinic": "SCB Medical College Eye Dept, Cuttack",
            "contact": "+91 671 240 7777 | scb@odi.gov.in"
        },
        "Normal": {
            "name": "Odisha Eye Health",
            "specialization": "Preventive Ophthalmology",
            "clinic": "Capital Hospital Eye OPD, Bhubaneswar",
            "contact": "Helpline: 104"
        }
    },
    "Punjab": {
        "Diabetic Retinopathy": {
            "name": "Dr. Harpreet Kaur",
            "specialization": "Retina Specialist",
            "clinic": "PGIMER Eye Dept, Chandigarh",
            "contact": "+91 172 275 5555 | pgimer@chd.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Gurpreet Singh",
            "specialization": "Glaucoma Consultant",
            "clinic": "Max Eye Care, Mohali",
            "contact": "+91 172 423 1234 | mohali@maxhealthcare.in"
        },
        "Cataract": {
            "name": "Dr. Amrit Pal Grewal",
            "specialization": "Cataract & Refractive Surgeon",
            "clinic": "Centre for Sight, Ludhiana",
            "contact": "+91 161 502 3456 | ludhiana@centreforsight.net"
        },
        "Normal": {
            "name": "Punjab Eye Health",
            "specialization": "Primary Ophthalmology",
            "clinic": "Civil Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Rajasthan": {
        "Diabetic Retinopathy": {
            "name": "Dr. Suresh Soni",
            "specialization": "Vitreo-Retinal Surgeon",
            "clinic": "SMS Medical College Eye Dept, Jaipur",
            "contact": "+91 141 256 0291 | sms@raj.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Renu Joshi",
            "specialization": "Glaucoma Specialist",
            "clinic": "Ratan Eye Hospital, Jaipur",
            "contact": "+91 141 274 3456 | info@rataneye.com"
        },
        "Cataract": {
            "name": "Dr. Mahesh Mathur",
            "specialization": "Cataract Surgeon",
            "clinic": "Apex Eye Centre, Jodhpur",
            "contact": "+91 291 261 2222 | apex@jodhpur.in"
        },
        "Normal": {
            "name": "Rajasthan Eye Health",
            "specialization": "Community Ophthalmology",
            "clinic": "JLN Hospital Eye OPD, Ajmer",
            "contact": "Helpline: 104"
        }
    },
    "Sikkim": {
        "Diabetic Retinopathy": {
            "name": "Dr. Tshering Lachungpa",
            "specialization": "Retina Consultant",
            "clinic": "STNM Hospital Eye Dept, Gangtok",
            "contact": "+91 3592 202 319 | stnm@sik.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Pemba Sherpa",
            "specialization": "Glaucoma Specialist",
            "clinic": "Sir Thutob Hospital Eye Unit",
            "contact": "+91 3592 202 555 | sth@sik.in"
        },
        "Cataract": {
            "name": "Dr. Dorjee Wangchuk",
            "specialization": "IOL Surgeon",
            "clinic": "Gangtok Eye Clinic",
            "contact": "+91 3592 204 789 | gangtokec@sik.in"
        },
        "Normal": {
            "name": "Sikkim Eye Health",
            "specialization": "Primary Eye Care",
            "clinic": "District Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Tamil Nadu": {
        "Diabetic Retinopathy": {
            "name": "Dr. S. Venkat",
            "specialization": "Vitreo-Retinal Surgeon",
            "clinic": "Sankara Nethralaya, Chennai",
            "contact": "+91 44 2827 1616 | appointment@snmail.org"
        },
        "Glaucoma": {
            "name": "Dr. Lakshmi Prasad",
            "specialization": "Glaucoma Consultant",
            "clinic": "Aravind Eye Hospital, Madurai",
            "contact": "+91 452 435 6100 | madurai@aravind.org"
        },
        "Cataract": {
            "name": "Dr. R. Murugesan",
            "specialization": "Phaco Specialist",
            "clinic": "Vasan Eye Care, Chennai",
            "contact": "+91 44 4340 0000 | info@vasaneye.in"
        },
        "Normal": {
            "name": "TN Health Services",
            "specialization": "Primary Eye Care",
            "clinic": "Regional Institute of Ophthalmology",
            "contact": "Toll Free: 104"
        }
    },
    "Telangana": {
        "Diabetic Retinopathy": {
            "name": "Dr. Ravi Kiran",
            "specialization": "Vitreoretinal Surgeon",
            "clinic": "LV Prasad Eye Institute, Hyderabad",
            "contact": "+91 40 3061 2345 | hyd@lvpei.org"
        },
        "Glaucoma": {
            "name": "Dr. Madhuri Reddy",
            "specialization": "Glaucoma Specialist",
            "clinic": "Pushpagiri Eye Institute, Hyderabad",
            "contact": "+91 40 2355 8899 | info@pushpagirieyeinstitute.com"
        },
        "Cataract": {
            "name": "Dr. Srinath Rao",
            "specialization": "Phaco & LASIK Surgeon",
            "clinic": "Sarojini Devi Eye Hospital",
            "contact": "+91 40 2314 7894 | sdeh@tg.gov.in"
        },
        "Normal": {
            "name": "Telangana Eye Health",
            "specialization": "Preventive Eye Care",
            "clinic": "Govt Eye Hospital, Hyderabad",
            "contact": "Helpline: 104"
        }
    },
    "Tripura": {
        "Diabetic Retinopathy": {
            "name": "Dr. Pradip Bhattacharjee",
            "specialization": "Retina Specialist",
            "clinic": "GBP Hospital Eye Dept, Agartala",
            "contact": "+91 381 241 2345 | gbp@tripura.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Suparna Saha",
            "specialization": "Glaucoma Consultant",
            "clinic": "Tripura Medical College Eye Dept",
            "contact": "+91 381 234 5679 | tmc@tri.in"
        },
        "Cataract": {
            "name": "Dr. Bimal Roy",
            "specialization": "Cataract Surgeon",
            "clinic": "Agartala Eye Care Centre",
            "contact": "+91 381 222 3456 | agartalaec@tri.in"
        },
        "Normal": {
            "name": "Tripura Eye Services",
            "specialization": "Community Ophthalmology",
            "clinic": "District Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Uttar Pradesh": {
        "Diabetic Retinopathy": {
            "name": "Dr. Rajendra Singh",
            "specialization": "Vitreoretinal Surgeon",
            "clinic": "SGPGI Eye Dept, Lucknow",
            "contact": "+91 522 266 8000 | sgpgi@up.gov.in"
        },
        "Glaucoma": {
            "name": "Dr. Alka Agarwal",
            "specialization": "Glaucoma Specialist",
            "clinic": "KGMU Eye Dept, Lucknow",
            "contact": "+91 522 225 7450 | kgmu@up.gov.in"
        },
        "Cataract": {
            "name": "Dr. Pradeep Sharma",
            "specialization": "Cataract & IOL Surgeon",
            "clinic": "Vasan Eye Care, Kanpur",
            "contact": "+91 512 234 0000 | kanpur@vasaneye.in"
        },
        "Normal": {
            "name": "UP Eye Health",
            "specialization": "Primary Eye Care",
            "clinic": "District Hospital Eye OPD",
            "contact": "Helpline: 104"
        }
    },
    "Uttarakhand": {
        "Diabetic Retinopathy": {
            "name": "Dr. Hemant Rawat",
            "specialization": "Retina Specialist",
            "clinic": "AIIMS Rishikesh Eye Dept",
            "contact": "+91 135 245 2222 | aiims.rishikesh@nic.in"
        },
        "Glaucoma": {
            "name": "Dr. Geeta Bisht",
            "specialization": "Glaucoma Consultant",
            "clinic": "HNB Base Hospital Eye Dept, Srinagar Garhwal",
            "contact": "+91 1346 252 456 | hnb@uk.gov.in"
        },
        "Cataract": {
            "name": "Dr. Arvind Joshi",
            "specialization": "Phaco Surgeon",
            "clinic": "Sudha Eye Hospital, Dehradun",
            "contact": "+91 135 275 1234 | sudha@dehradun.in"
        },
        "Normal": {
            "name": "Uttarakhand Eye Health",
            "specialization": "Community Eye Care",
            "clinic": "Doon Hospital Eye OPD, Dehradun",
            "contact": "Helpline: 104"
        }
    },
    "West Bengal": {
        "Diabetic Retinopathy": {
            "name": "Dr. Abhijit Das",
            "specialization": "Retina Specialist",
            "clinic": "Susrut Eye Foundation, Kolkata",
            "contact": "+91 33 2419 6223 | support@susrut.org"
        },
        "Glaucoma": {
            "name": "Dr. Moumita Ghosh",
            "specialization": "Glaucoma Division",
            "clinic": "B B Eye Foundation",
            "contact": "+91 33 2280 1234 | contact@bbeye.com"
        },
        "Cataract": {
            "name": "Dr. Sourav Mondal",
            "specialization": "Cataract & IOL",
            "clinic": "Apollo Clinic Eye Care",
            "contact": "+91 33 4022 4022 | kolkata@apolloclinic.com"
        },
        "Normal": {
            "name": "Kolkata Vision Center",
            "specialization": "General Optometry",
            "clinic": "Govt Ophthalmic Unit, Kolkata",
            "contact": "Helpline: 033-1234"
        }
    },
}

DEFAULT_STATE = "Delhi"


def get_doctor_details(state: str, disease: str) -> Dict[str, str]:
    """Retrieve doctor details for a given state and disease."""
    # Fallback to DEFAULT_STATE if specified state is not found
    state_data = DOCTORS_BY_STATE.get(state, DOCTORS_BY_STATE[DEFAULT_STATE])
    # Fallback to "Normal" if disease is not found in the state mapping
    return state_data.get(disease, state_data["Normal"])


def get_all_doctor_records(disease: str) -> List[Dict[str, str]]:
    """
    Return a list of doctor records for ALL states for the given disease.
    Each record includes the state name plus doctor details.
    """
    records = []
    for state in sorted(DOCTORS_BY_STATE.keys()):
        state_data = DOCTORS_BY_STATE[state]
        doc = state_data.get(disease, state_data.get("Normal", {}))
        records.append({
            "state": state,
            **doc
        })
    return records


def get_available_states() -> List[str]:
    """Returns list of states we have doctor data for."""
    return sorted(list(DOCTORS_BY_STATE.keys()))
