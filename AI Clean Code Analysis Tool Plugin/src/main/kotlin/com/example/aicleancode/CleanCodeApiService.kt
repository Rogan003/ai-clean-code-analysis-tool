package com.example.aicleancode

import com.google.gson.Gson
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import java.io.IOException

data class MethodRequest(val code_snippet: String)
data class ClassRequest(val code_snippet: String, val average_method_score: Double?)
data class PredictionResponse(
    val prediction: Int,
    val prediction_label: String,
    val confidence: Double
)

class CleanCodeApiService(private val baseUrl: String = "http://localhost:8000") {
    private val client = OkHttpClient()
    private val gson = Gson()
    private val jsonMediaType = "application/json; charset=utf-8".toMediaType()

    fun predictMethod(codeSnippet: String): PredictionResponse? {
        val request = MethodRequest(codeSnippet)
        val json = gson.toJson(request)
        val body = json.toRequestBody(jsonMediaType)

        val httpRequest = Request.Builder()
            .url("$baseUrl/predict/method")
            .post(body)
            .build()

        return try {
            client.newCall(httpRequest).execute().use { response ->
                if (!response.isSuccessful) {
                    println("Failed to predict method: ${response.code}")
                    return null
                }
                val responseBody = response.body?.string() ?: return null
                gson.fromJson(responseBody, PredictionResponse::class.java)
            }
        } catch (e: IOException) {
            println("Error calling API: ${e.message}")
            null
        }
    }

    fun predictClass(codeSnippet: String, averageMethodScore: Double? = null): PredictionResponse? {
        val request = ClassRequest(codeSnippet, averageMethodScore)
        val json = gson.toJson(request)
        val body = json.toRequestBody(jsonMediaType)

        val httpRequest = Request.Builder()
            .url("$baseUrl/predict/class")
            .post(body)
            .build()

        return try {
            client.newCall(httpRequest).execute().use { response ->
                if (!response.isSuccessful) {
                    println("Failed to predict class: ${response.code}")
                    return null
                }
                val responseBody = response.body?.string() ?: return null
                gson.fromJson(responseBody, PredictionResponse::class.java)
            }
        } catch (e: IOException) {
            println("Error calling API: ${e.message}")
            null
        }
    }
}
