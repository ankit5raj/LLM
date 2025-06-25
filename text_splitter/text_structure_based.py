from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """
Modern-day Indore traces its roots to its 16th-century founding as a trading hub between the Deccan and Delhi.[23] It was founded on the banks of the Kanh and Saraswati rivers.[24] The city came under the Maratha Empire, on 18 May 1724, after Peshwa Baji Rao I assumed the full control of Malwa.[25] During the days of the British Raj, Indore State was a 19 Gun Salute (21 locally) princely state (a rare high rank) ruled by the Maratha Holkar dynasty, until they acceded to the Union of India.[26]

Indore functions as the financial capital of Madhya Pradesh and was home to the Madhya Pradesh Stock Exchange till its derecognition in 2015.

Indore has been selected as one of the 100 Indian cities to be developed as a smart city under the Smart Cities Mission.[27] It also qualified in the first round of Smart Cities Mission and was selected as one of the first twenty cities to be developed as Smart Cities.[28] Indore has been part of the Swachh Survekshan since its inception and had ranked 25th in 2016.[29] It has been ranked as India's cleanest city seven years in a row as per the Swachh Survekshan for the years 2017, 2018, 2019, 2020, 2021, 2022 and 2023.[30][31][32][33][34] Meanwhile, Indore has also been declared as India's first 'water plus' city under the Swachhta Survekshan 2021. Indore became the only Indian city to be selected for International Clean Air Catalyst Programme. The project, with cooperation of the Indore Municipal Corporation and the Madhya Pradesh Pollution Control Board, will be operated for a period of five years to purify the air in the city. Indore will penalise anyone giving alms to beggars starting 1 January 2025, expanding a previous ban on giving to child beggars. This initiative aims to eradicate begging, with officials claiming it disrupts the begging cycle.[35]
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 0
)

result = splitter.split_text(text)

print(result[1])