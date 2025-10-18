package com.example.aicleancode

import com.intellij.ide.BrowserUtil
import com.intellij.notification.Notification
import com.intellij.notification.NotificationType
import com.intellij.notification.Notifications
import com.intellij.openapi.actionSystem.*
import com.intellij.openapi.application.ApplicationManager
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.markup.EffectType
import com.intellij.openapi.editor.markup.HighlighterLayer
import com.intellij.openapi.editor.markup.HighlighterTargetArea
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.ui.Messages
import com.intellij.psi.PsiManager
import com.intellij.ui.JBColor
import java.awt.Color
import java.awt.Font

class RunCleanCodeAnalysisAction : AnAction("AI Clean Code Analysis"), DumbAware {

    override fun update(e: AnActionEvent) {
        val project = e.project
        val editor = e.getData(CommonDataKeys.EDITOR)
        val file = e.getData(CommonDataKeys.VIRTUAL_FILE)
        val enabled = project != null && editor != null && file?.fileType?.name == "JAVA"
        e.presentation.isEnabledAndVisible = true
        e.presentation.isEnabled = enabled
        e.presentation.description = "Analyze the currently opened Java file for clean code issues (demo)"
    }

    override fun actionPerformed(e: AnActionEvent) {
        val project = e.project ?: return
        val editor = e.getRequiredData(CommonDataKeys.EDITOR)
        val vFile = e.getRequiredData(CommonDataKeys.VIRTUAL_FILE)

        if (vFile.fileType.name != "JAVA") {
            Messages.showInfoMessage(project, "Please open a Java file to run the AI Clean Code Analysis.", "AI Clean Code Analysis")
            return
        }

        // Clear previous highlighters
        DemoHighlighters.removeForEditor(editor)

        // Show notification that analysis is starting
        Notifications.Bus.notify(
            Notification(
                "AI Clean Code Analysis",
                "AI Clean Code Analysis Started",
                "Analyzing ${vFile.name}...",
                NotificationType.INFORMATION
            ),
            project
        )

        // Run analysis in background thread
        ApplicationManager.getApplication().executeOnPooledThread {
            try {
                val psiFile = PsiManager.getInstance(project).findFile(vFile) ?: return@executeOnPooledThread
                val document = editor.document

                // Extract methods and classes
                val extractor = JavaCodeExtractor()
                val elements = extractor.extractMethodsAndClass(psiFile, document)

                // Create API service
                val apiService = CleanCodeApiService()

                // Track method predictions per class for class analysis
                var currentClassMethodPredictions = mutableListOf<Int>()

                // Analyze each element in order; methods should come before their class (as ensured by JavaCodeExtractor)
                for (element in elements) {
                    when (element.type) {
                        ElementType.METHOD -> {
                            val prediction = apiService.predictMethod(element.code)
                            if (prediction != null) {
                                currentClassMethodPredictions.add(prediction.prediction)
                                // Highlight method on UI thread
                                ApplicationManager.getApplication().invokeLater {
                                    highlightElement(editor, element, prediction.prediction)
                                }
                            }
                        }
                        ElementType.CLASS -> {
                            // Calculate average method score for the just-seen methods belonging to this class
                            val avgScore = if (currentClassMethodPredictions.isNotEmpty()) {
                                extractor.calculateAverageMethodScore(currentClassMethodPredictions)
                            } else null

                            val prediction = apiService.predictClass(element.code, avgScore)
                            if (prediction != null) {
                                // Highlight class on UI thread
                                ApplicationManager.getApplication().invokeLater {
                                    highlightElement(editor, element, prediction.prediction)
                                }
                            }

                            // Reset for next class
                            currentClassMethodPredictions = mutableListOf()
                        }
                    }
                }

                // Show completion notification
                ApplicationManager.getApplication().invokeLater {
                    Notifications.Bus.notify(
                        Notification(
                            "AI Clean Code Analysis",
                            "AI Clean Code Analysis Finished",
                            "Analysis complete for ${vFile.name}",
                            NotificationType.INFORMATION
                        ),
                        project
                    )
                }

            } catch (ex: Exception) {
                ApplicationManager.getApplication().invokeLater {
                    Notifications.Bus.notify(
                        Notification(
                            "AI Clean Code Analysis",
                            "AI Clean Code Analysis Failed",
                            "Error: ${ex.message}. Make sure the API server is running on http://localhost:8000",
                            NotificationType.ERROR
                        ),
                        project
                    )
                }
            }
        }
    }

    private fun highlightElement(editor: Editor, element: CodeElement, prediction: Int) {
        // Define fill attributes
        val green = TextAttributes(null, JBColor(Color(232, 245, 233), Color(39, 51, 43)), JBColor.GREEN.darker(), EffectType.BOXED, Font.BOLD)
        val yellow = TextAttributes(null, JBColor(Color(255, 243, 205), Color(58, 50, 35)), JBColor.YELLOW.darker(), EffectType.BOXED, Font.PLAIN)
        val red = TextAttributes(null, JBColor(Color(255, 235, 238), Color(63, 40, 43)), JBColor.RED, EffectType.BOXED, Font.BOLD)

        // Define stripe attributes
        val errorStripeGreen = TextAttributes(null, null, JBColor(0x43A047, 0x43A047), EffectType.LINE_UNDERSCORE, Font.BOLD)
        val errorStripeYellow = TextAttributes(null, null, JBColor(0xFFC107, 0xFFC107), EffectType.LINE_UNDERSCORE, Font.PLAIN)
        val errorStripeRed = TextAttributes(null, null, JBColor(0xE53935, 0xE53935), EffectType.LINE_UNDERSCORE, Font.BOLD)

        val (fillAttrs, stripeAttrs, layer) = when (prediction) {
            0 -> Triple(green, errorStripeGreen, HighlighterLayer.ADDITIONAL_SYNTAX)
            1 -> Triple(yellow, errorStripeYellow, HighlighterLayer.WARNING)
            else -> Triple(red, errorStripeRed, HighlighterLayer.ERROR)
        }

        val highlighters = highlightLines(editor, element.startLine, element.endLine, fillAttrs, stripeAttrs, layer)
        DemoHighlighters.addForEditor(editor, highlighters)
    }

    // Helper to highlight a 1-based line range with both fill and stripe markers
    private fun highlightLines(
        editor: Editor,
        startLine: Int,
        endLine: Int,
        fillAttrs: TextAttributes,
        stripeAttrs: TextAttributes,
        stripeLayer: Int
    ): List<com.intellij.openapi.editor.markup.RangeHighlighter> {
        val doc = editor.document
        fun clampLine(line: Int) = line.coerceIn(1, doc.lineCount) // 1-based input
        val startOffset = doc.getLineStartOffset(clampLine(startLine) - 1)
        val endOffset = doc.getLineEndOffset(clampLine(endLine) - 1)

        val markup = editor.markupModel
        val fillHl = markup.addRangeHighlighter(startOffset, endOffset, HighlighterLayer.SELECTION - 1, fillAttrs, HighlighterTargetArea.EXACT_RANGE)
        val stripeHl = markup.addRangeHighlighter(startOffset, endOffset, stripeLayer, stripeAttrs, HighlighterTargetArea.EXACT_RANGE)
        return listOf(fillHl, stripeHl)
    }

    private fun openDocs() {
        BrowserUtil.browse("https://plugins.jetbrains.com/docs/intellij/welcome.html")
    }
}

private object DemoHighlighters {
    private val key = "ai.clean.code.demo.highlighters"

    fun removeForEditor(editor: Editor) {
        val userData = editor.getUserData(EditorHighlighterKey)
        userData?.forEach { it.dispose() }
        editor.putUserData(EditorHighlighterKey, null)
    }

    fun addForEditor(editor: Editor, list: List<com.intellij.openapi.editor.markup.RangeHighlighter>) {
        val existing = editor.getUserData(EditorHighlighterKey)?.toMutableList() ?: mutableListOf()
        existing.addAll(list)
        editor.putUserData(EditorHighlighterKey, existing)
    }

    private val EditorHighlighterKey = com.intellij.openapi.util.Key.create<MutableList<com.intellij.openapi.editor.markup.RangeHighlighter>>(key)
}
