import pandas as pd
import time
from preprocessing.batch_processing_today import *
from preprocessing.batch_processing_profiles import *
from preprocessing.get_content_from_excel_url import *

import openai

# Define a custom exception
class OpenAIRateLimitError(Exception):
    def __init__(self, message="Server is busy.Please retry after few minutes"):
        super().__init__(message)
#------------------------------------------------------------------------------------------------------
# def get_results(rr_df, bench_df):
def get_results(rr_df, bench_df,isCvSkills):

    #--------------------------------------------------------------------------------------------------
    # Preprocessing File
    try:
        from preprocessing.process_rr_details import clean_rr_overview_text,get_rr_skills_from_overview

        # ########### Process RR ###################
        start_time = time.time()
        process_df = clean_rr_overview_text(rr_df)
        process_df = get_rr_skills_from_overview(process_df)
        process_df["RR"] = process_df["RR"].astype(str)
        process_df = pd.read_excel(r"assets/output/RR_Processed_skills.xlsx")
        end_time = time.time()

        preprocessing_time = (end_time-start_time)/60

        start_time = time.time()
        model = llm()
        # sharepoint_data = get_folder_file(c_id, c_secret, s_url, r_url)
        raw_data = read_file(bench_df)
        print(raw_data)
        print("isCvSkills : ",isCvSkills)
        if isCvSkills:
            primary_data = combine_skills(raw_data)
            print("Skills_df",primary_data)
            print("Creating raw skills json")
            raw_skill_json = save_and_read_data(primary_data, r"output_files\\json\\raw_skills.json")
            print('Creating Raw Skills JSON')
            for item in raw_skill_json:
                item['raw_skills'] = ["" if skill == "Not Available" else skill for skill in item['raw_skills']]
            cv_cleaned_skills_output = cv_process_in_batches(data = raw_skill_json)
            print('Creating Cleaned Skills JSON')
            with open(r"output_files/json/cleaned_skills.json", 'w') as json_file:
                json.dump(cv_cleaned_skills_output, json_file, indent=4)
            convert_to_dataFrame(r"output_files/json/cleaned_skills.json",bench_df=raw_data)
            print("created cleaned skills json")
            profile_skill = pd.read_excel(r'output_files/excel/datafarme_output_v1.xlsx')
            profile_skill_df,empty_profile_skill_df = split_dataframe(profile_skill)
            empty_profile_skill_df.to_excel(r'output_files/excel/empty_skills_output.xlsx', index=False)
            # profile_skill_df = pd.read_excel(r'output_files/excel/datafarme_output_v1.xlsx')
            # read_json_df = read_cleaned_skills() #returns json data 
            print("xls created")
            print("="*80)
            print("profile_skill_df")
            print(profile_skill_df)
            print("="*80)
        else:
            sharepoint_data =  call_functions(raw_data, 'Sharepoint_url', c_id, c_secret)
            data = read_file(sharepoint_data)
            print(data)
            data['raw_skills'] = data['raw_skills'].apply(add_brackets)
            data['raw_skills'] = data['raw_skills'].apply(convert_to_list)
            raw_skill_json = save_and_read_data(data, r"output_files/json/raw_skills.json")
            cleaned_raw_skill_json = clean_raw_skills(raw_skill_json)

            response = chat_completion_to_clean_skillset(raw_skill_json,cleaned_raw_skill_json, model)

            read_json_df = read_cleaned_skills() #returns json data 

            # five profiles are going in to gpt to extract the clean skills
            batch_size = 5
            summaries = []
            i = 1
            for start in range(0, len(read_json_df), batch_size):
                print(i,"/",batch_size)
                end = start + batch_size
                chunk = read_json_df[start:end]
                combined_prompt_clean_skillset = combined_prompt_cleaned_skills(chunk)
                summary_response = chat_completion_to_summarize_skillset(combined_prompt_clean_skillset, model)

                print("-"*80)
                print("Length of each batch size: ", len(summary_response))
                print(summary_response)
                print("-"*80)
                summaries.append(summary_response)
                i += 1
            # print(len(summaries))
            # print(summaries)

            batch_process(summaries, batch_size, raw_skill_json)
            convert_to_dataFrame(r"output_files/json/cleaned_skills.json",bench_df)
            
            profile_skill_df = pd.read_excel(r'output_files/excel/datafarme_output_v1.xlsx')

            print("="*80)
            print(profile_skill_df.head())
            print("="*80)


        #-----------

        print(profile_skill_df.columns)
        print("profile_skill_df.columns")
        profile_skill_reduced_df = profile_skill_df.copy()
        profile_skill_reduced_df.rename(columns={'PID': 'portal_id',
                                                    'EE Name':'Employee Name',
                                                    'raw_skills':'Raw Skills',
                                                    'bench_period':'bench_period',
                                                    'summary_extracted_skills':'Skill_summary',
                                                    'cleaned_extracted_skills': "Skills"
                                                    }, inplace=True)
        print(profile_skill_reduced_df.columns)
        print("profile_skill_reduced_df.columns *****************")
        print(profile_skill_reduced_df.head())

        end_time = time.time()

        skill_extraction_time = (end_time-start_time)/60


        #--------------------------------------------------------------------------------------------------
        # Generating Embeddings File

        from recommendation.generate_embeddings import generate_embedding_for_dataframe
        start_time = time.time()
        # Generate and Dump Embeddings 
        generate_embedding_for_dataframe(process_df, profile_skill_reduced_df)
        print("Done with Generating Embeddings !!!!!")

        end_time = time.time()

        embedding_time = (end_time-start_time)/60

        #--------------------------------------------------------------------------------------------------
        # Generating Recommendations for RR's

        start_time = time.time()

        processed_rr= pd.read_csv("assets/data/embeddings/embedded_rr_details.csv")
        processed_cv= pd.read_csv("assets/data/embeddings/embedded_cv_details.csv")
        from recommendation.generate_recommendation_for_rrs import profile_recommender

        # Case 1: Recommend Profiles for each RR
        profile_recommendation_df = profile_recommender(processed_rr,processed_cv)
        profile_recommendation_df.to_excel("assets/output/RR_To_Profiles_Recommendations.xlsx", index=False)
        print("Generated Recommendations for RRs")

        #--------------------------------------------------------------------------------------------------
        # Generating Recommendations for Profiles's
        processed_rr= pd.read_csv("assets/data/embeddings/embedded_rr_details.csv")
        processed_cv= pd.read_csv("assets/data/embeddings/embedded_cv_details.csv")
        from recommendation.generate_recommendations_for_profiles import rr_recommender
        # Case 3: Recommend RR for each profile
        rr_recommendation_df = rr_recommender(processed_rr,processed_cv)

        rr_recommendation_df.to_excel("assets/output/Profiles_To_RR_Recommendations.xlsx", index=False)
        print("Generated Recommendations for Profiles")

        end_time = time.time()

        generate_recommendations_time = (end_time-start_time)/60

        #--------------------------------------------------------------------------------------------------

        # Generating Refined Recommendations for each RR
        from recommendation.generate_refined_recommendations_for_top_rr import generate_refined_recommendations
        from recommendation.generate_refined_recommendations_for_profiles import generate_refined_recommendations_profiles

        start_time = time.time()

        profile_links_df = raw_data[['PID','Profile Link']]
        generate_refined_recommendations(profile_links_df)
        generate_refined_recommendations_profiles(profile_links_df)

        end_time = time.time()

        refined_recommendations_time = (end_time-start_time)/60

        print("+"*80)
        print(f"Preprocessing time: {preprocessing_time:.2f} minutes")
        print(f"Skill Extraction time: {skill_extraction_time:.2f} minutes")
        print(f"Embedding generation time: {embedding_time:.2f} minutes")
        print(f"Recommendation generation time: {generate_recommendations_time:.2f} minutes")
        print(f"Refined Recommendation generation time: {refined_recommendations_time:.2f} minutes")

        total_time = preprocessing_time + skill_extraction_time + embedding_time + generate_recommendations_time + refined_recommendations_time

        print(f"Total time taken for the entire backend is: {total_time:.2f} minutes")
        print("+"*80)
    except Exception as e:
        if "rate limit" in str(e).lower():  # Check for rate limit error in message
            raise OpenAIRateLimitError()

if __name__ == "__main__":
    pass
    # Loading The data
    # rr_df = pd.read_excel(r"assets\data\Global Demand - Matching POC - Demo.xlsx")
    # bench_df = pd.read_excel(r"assets\data\Global Bench for Matching POC - Demo.xlsx")
    # start_time = time.time()
    # get_results(rr_df, bench_df)
    # end_time = time.time()
    # total_time_minutes = (end_time - start_time) / 60

    # print(f"Total time taken for the entire backend is: {total_time_minutes:.2f} minutes") 