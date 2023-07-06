import { OpenAI } from "langchain/llms/openai";
import { FaissStore } from "langchain/vectorstores/faiss";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { loadQAStuffChain, loadQAMapReduceChain } from "langchain/chains";

import express from "express";
// import https from "https";
import { fileURLToPath } from "url";
import path, { dirname } from "path";
dotenv.config();
import * as dotenv from "dotenv";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);
const app = express();
const port = 4000;

// https.createServer(app).listen(process.env.PORT)

app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

app.get("/api/health", async (req, res) => {
  res.json({
    success: true,
    message: "Server is running on its best",
  });
});

app.get("/ask", async (req, res) => {
  try {
    const llmA = new OpenAI({ modelName: "gpt-3.5-turbo" });
    const chainA = loadQAStuffChain(llmA);
    const directory = process.env.DIR;

    const loadedVectorStore = await FaissStore.load(
      directory,
      new OpenAIEmbeddings()
    );

    const question = "give me summary of mind?";
    const result = await loadedVectorStore.similaritySearch(question, 1);
    const resA = await chainA.call({
      input_documents: result,
      question,
    });

    res.json({ result: resA });
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});
