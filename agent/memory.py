#persistent memory layer for the orion agent
#stores every agent run and enables semantic seach across past applications

import json
import chromadb
from chromadb.config import Settings

class AgentMemory:
    def __init__(self, path:str = "./memory_store"):
        #chromaDB stores data on disk at its path
        #unlike a regular dictionary, this survives restarts
        self.client = chromadb.PersistentClient(path=path)

        #collection is a table in a normal db
        #get_or_create_collection will load if exists, otherwise make new
        self.runs = self.client.get_or_create_collection(
            name="agent_runs",
            metadata={"hreader_space":"cosine"}
        )
    def save(self,trace: dict) -> None:
        # we store three things per run
        #1. document - text chromaDB uses to find similar runs later
        #2. metadata - structured data we want to filter (by score, date, etc.)
        #3. if- unique identifier for this run

        #document is what gets turned into a vector (embedding)
        #we combine job description + summary so search works on meaning
        final = trace.get("final_output", {})
        job_text = ""
        for step in trace.get("steps", []):
            if step["tool_name"] == "scpre_fit":
                job_text = str(step["tool_input"].get("job_description", ""))
                break
        document = f"""
        job: {job_text[:500]}
        Summary:{final.get("sumamry", "")}
        Matched skills: {",".join(final.get("matched_skills", []))}
        Skill gaps: {",".join(final.get("skill_gaps", []))}
        """.strip()

        self.runs.add(
            documents=[document],
            metadatas=[{
                "run_id": trace["run_id"],
                "started_at": trace["started_at"],
                "fit_score": int(final.get("fit_score") or 0),
                "status": trace["status"],
                "goal": trace["goal"]
            }],
            ids=[trace["run_id"]],
        )
        print(f"[memory] saved run {trace['run_id']}")

        def search(self,query: str, n_results: int=3) -> list:
            #convert query to a vector and find the n closest stored runs
            #returns the most semantically similar past applications
            results = self.runs.query(
                query_texts=[query],
                n_results=n_results,
            )

            #reformat into a clean list of dicts
            runs = []
            for i, meta in enumerate(results["metadatas"][0]):
                runs.append({
                    "run_id": meta["run_id"],
                    "fit_score": meta["fit_score"],
                    "started_at": meta["started_at"],
                    "similarity_rank": i + 1,
                    "document": results["documents"][0][i],
                })
                return runs
            def get_all(self) -> list:
                #returns every stored run- used by the dashboard to lsit history
                results = self.runs.get()
                return [
                    {
                        "run_id": meta["run_id"],
                        "fit_score": meta["fit_score"],
                        "started_at": meta["started_at"],
                        "goal": meta["goal"],
                    }
                    for meta in results["metadatas"]
                ]
#single global instance, same pattern as registry
memory = AgentMemory()