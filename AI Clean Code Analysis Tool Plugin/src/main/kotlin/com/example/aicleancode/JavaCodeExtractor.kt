package com.example.aicleancode

import com.intellij.openapi.editor.Document
import com.intellij.psi.*
import com.intellij.psi.util.PsiTreeUtil

data class CodeElement(
    val code: String,
    val startLine: Int,
    val endLine: Int,
    val type: ElementType
)

enum class ElementType {
    METHOD, CLASS
}

class JavaCodeExtractor {

    fun extractMethodsAndClass(psiFile: PsiFile, document: Document): List<CodeElement> {
        val elements = mutableListOf<CodeElement>()

        // Find all classes in the file
        val classes = PsiTreeUtil.findChildrenOfType(psiFile, PsiClass::class.java)

        for (psiClass in classes) {
            // Skip anonymous and inner classes for simplicity (or handle them if needed)
            if (psiClass.name == null) continue

            // Extract all methods from this class FIRST
            val methods = psiClass.methods
            for (method in methods) {
                val methodText = method.text
                val methodStartLine = document.getLineNumber(method.textRange.startOffset) + 1
                val methodEndLine = document.getLineNumber(method.textRange.endOffset) + 1
                elements.add(CodeElement(methodText, methodStartLine, methodEndLine, ElementType.METHOD))
            }

            // Then add the class element so that callers can compute method averages before class-level call
            val classText = psiClass.text
            val classStartLine = document.getLineNumber(psiClass.textRange.startOffset) + 1
            val classEndLine = document.getLineNumber(psiClass.textRange.endOffset) + 1
            elements.add(CodeElement(classText, classStartLine, classEndLine, ElementType.CLASS))
        }

        return elements
    }

    fun calculateAverageMethodScore(methodPredictions: List<Int>): Double {
        if (methodPredictions.isEmpty()) return 1.0
        return methodPredictions.average()
    }
}
