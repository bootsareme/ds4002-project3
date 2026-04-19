#imports
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import torch
import matplotlib.pyplot as plt

#read in files
references = pd.read_excel("../DATA/meme_compilation.xlsx")
results_blip = pd.read_csv("../OUTPUT/results_blip.csv")
results_blipv2 = pd.read_csv("../OUTPUT/results_blipv2.csv")

#perform SBERT analysis
model = SentenceTransformer("all-MiniLM-L6-v2")

summary_rows = []
all_rows = []

query_files = [("blip", results_blip), ("blipv2", results_blipv2)]

for col in references.columns:

    sent_list = references[col].dropna().tolist()
    sent_emb = model.encode(sent_list, convert_to_tensor=True)

    for source_name, qdf in query_files:

        queries = qdf[col].dropna().tolist()
        query_emb = model.encode(queries, convert_to_tensor=True)

        sim_matrix = util.cos_sim(query_emb, sent_emb)

        for q_idx, query in enumerate(queries):

            scores = sim_matrix[q_idx]

            #add to summary file
            summary_rows.append({"query_source": source_name, "meme": col, "query": query, 
                                 "average_score": float(torch.mean(scores)), "min_score": float(torch.min(scores)), 
                                 "max_score": float(torch.max(scores))})

            #add everything to results
            for s_idx, score in enumerate(scores):
                all_rows.append({"query_source": source_name, "meme": col, "query": query,
                                 "sentence": sent_list[s_idx], "score": float(score)})

#save files
all_results_df = pd.DataFrame(all_rows)
summary_df = pd.DataFrame(summary_rows)

all_results_df.to_csv("../OUTPUT/all_similarity_scores.csv", index=False)
summary_df.to_csv("../OUTPUT/query_similarity_summary.csv", index=False)

#now for making plots
summary = pd.read_csv("../OUTPUT/query_similarity_summary.csv")
overall = summary.groupby("query_source")["average_score"].mean()
column_comparison = summary.groupby(["query_source","meme"])["average_score"].mean().unstack()
column_avg = summary.groupby(["query_source","meme"])["average_score"].mean().reset_index()

q1 = column_avg[column_avg["query_source"] == "blip"]
q2 = column_avg[column_avg["query_source"] == "blipv2"]


#plot diference in total average model performance
overall.plot(kind="bar", color="#95beed")
ax = plt.gca()

plt.grid(axis="y", linestyle="--", alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=45)

plt.xlabel("Model")
plt.ylabel("Average Similarity Score")
plt.title("Overall Model Performance")

plt.savefig('../OUTPUT/total_avg_comparison.png')
plt.show()


#plot comparison of average scores for each meme
column_comparison.plot(kind="bar", colormap="coolwarm")
ax = plt.gca()

plt.grid(axis="y", linestyle="--", alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=45)

plt.ylabel("Average Similarity")
plt.title("Average Scores for each Meme")

plt.savefig('../OUTPUT/avg_comparison.png')
plt.show()


#make individual plots to compare average of each meme for both models
plt.figure()
ax = plt.gca()

plt.bar(q1["meme"], q1["average_score"], color="#95beed")

plt.xlabel("Meme")
plt.ylabel("Average Similarity Score")
plt.title("Average Scores for each Meme (blip)")

plt.grid(axis="y", linestyle="--", alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=45)

plt.savefig('../OUTPUT/avg_scores_blip.png')

plt.show()


plt.figure()
ax = plt.gca()

plt.bar(q2["meme"], q2["average_score"], color="#95beed")

plt.xlabel("Meme")
plt.ylabel("Average Similarity Score")
plt.title("Average Scores for each Meme (blipv2)")

plt.grid(axis="y", linestyle="--", alpha=0.7)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=45)

plt.savefig('../OUTPUT/avg_scores_blipv2.png')

plt.show()