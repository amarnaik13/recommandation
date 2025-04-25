import pandas as pd
import time
import openai
import sys
import os
import json  # ✅ Fix: Added missing import

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# ✅ Fix: Removed duplicate import and ensured proper import paths
from preprocessing.batch_processing_today import *
from preprocessing.batch_processing_profiles import *
from preprocessing.get_content_from_excel_url import *
from preprocessing.process_rr_details import clean_rr_overview_text, get_rr_skills_from_overview
from recommendation.generate_embeddings import generate_embedding_for_dataframe
from recommendation.generate_recommendation_for_rrs import profile_recommender
from recommendation.generate_recommendations_for_profiles import rr_recommender
from recommendation.generate_refined_recommendations_for_top_rr import generate_refined_recommendations
from recommendation.generate_refined_recommendations_for_profiles import generate_refined_recommendations_profiles

# ✅ Fix: Ensure `llm` and `read_cleaned_skills` are correctly imported
from some_module import llm, read_cleaned_skills  

class OpenAIRateLimitError(Exception):
    def __init__(self, message="Server is busy. Please retry after a few minutes"):
        super().__init__(message)

def get_results(rr_df, bench_df, isCvSkills):
    try:
        start_time = time.time()

        # ✅ Fix: Remove unnecessary reloading of DataFrame
        process_df = clean_rr_overview_text(rr_df)
        process_df = get_rr_skills_from_overview(process_df)
        process_df["RR"] = process_df["RR"].astype(str)

        preprocessing_time = (time.time() - start_time) / 60

        start_time = time.time()
        model = llm()
        raw_data = read_file(bench_df)

        print("isCvSkills : ", isCvSkills)
        if isCvSkills:
            primary_data = combine_skills(raw_data)
            raw_skill_json = save_and_read_data(primary_data, r"output_files/json/raw_skills.json")

            for item in raw_skill_json:
                item['raw_skills'] = ["" if skill == "Not Available" else skill for skill in item['raw_skills']]
                
            cv_cleaned_skills_output = cv_process_in_batches(data=raw_skill_json)

            with open(r"output_files/json/cleaned_skills.json", 'w') as json_file:
                json.dump(cv_cleaned_skills_output, json_file, indent=4)

            convert_to_dataFrame(r"output_files/json/cleaned_skills.json", bench_df=raw_data)
            profile_skill_df, empty_profile_skill_df = split_dataframe(pd.read_excel(r'output_files/excel/datafarme_output_v1.xlsx'))
            empty_profile_skill_df.to_excel(r'output_files/excel/empty_skills_output.xlsx', index=False)

        else:
            sharepoint_data = call_functions(raw_data, 'Sharepoint_url', c_id, c_secret)
            data = read_file(sharepoint_data)
            raw_skill_json = save_and_read_data(data, r"output_files/json/raw_skills.json")
            cleaned_raw_skill_json = clean_raw_skills(raw_skill_json)

            response = chat_completion_to_clean_skillset(raw_skill_json, cleaned_raw_skill_json, model)
            read_json_df = read_cleaned_skills()

            batch_size = 5
            summaries = []
            for start in range(0, len(read_json_df), batch_size):
                chunk = read_json_df[start:start + batch_size]
                combined_prompt_clean_skillset = combined_prompt_cleaned_skills(chunk)
                summaries.append(chat_completion_to_summarize_skillset(combined_prompt_clean_skillset, model))

            batch_process(summaries, batch_size, raw_skill_json)
            convert_to_dataFrame(r"output_files/json/cleaned_skills.json", bench_df)

            profile_skill_df = pd.read_excel(r'output_files/excel/datafarme_output_v1.xlsx')

        profile_skill_reduced_df = profile_skill_df.rename(columns={
            'PID': 'portal_id',
            'EE Name': 'Employee Name',
            'raw_skills': 'Raw Skills',
            'bench_period': 'bench_period',
            'summary_extracted_skills': 'Skill_summary',
            'cleaned_extracted_skills': "Skills"
        })

        skill_extraction_time = (time.time() - start_time) / 60

        # ✅ Fix: Ensure embedding functions are called correctly
        start_time = time.time()
        generate_embedding_for_dataframe(process_df, profile_skill_reduced_df)
        embedding_time = (time.time() - start_time) / 60

        start_time = time.time()
        processed_rr = pd.read_csv("assets/data/embeddings/embedded_rr_details.csv")
        processed_cv = pd.read_csv("assets/data/embeddings/embedded_cv_details.csv")

        profile_recommender(processed_rr, processed_cv).to_excel("assets/output/RR_To_Profiles_Recommendations.xlsx", index=False)
        rr_recommender(processed_rr, processed_cv).to_excel("assets/output/Profiles_To_RR_Recommendations.xlsx", index=False)

        generate_recommendations_time = (time.time() - start_time) / 60

        start_time = time.time()
        profile_links_df = raw_data[['PID', 'Profile Link']]
        generate_refined_recommendations(profile_links_df)
        generate_refined_recommendations_profiles(profile_links_df)
        refined_recommendations_time = (time.time() - start_time) / 60

    except Exception as e:
        if isinstance(e, openai.error.RateLimitError):
            raise OpenAIRateLimitError()
        else:
            raise  
