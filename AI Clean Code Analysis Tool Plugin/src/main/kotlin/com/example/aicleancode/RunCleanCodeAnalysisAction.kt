package com.example.aicleancode

import com.intellij.ide.BrowserUtil
import com.intellij.notification.Notification
import com.intellij.notification.NotificationType
import com.intellij.notification.Notifications
import com.intellij.openapi.actionSystem.*
import com.intellij.openapi.editor.Editor
import com.intellij.openapi.editor.colors.EditorColors
import com.intellij.openapi.editor.markup.EffectType
import com.intellij.openapi.editor.markup.HighlighterLayer
import com.intellij.openapi.editor.markup.HighlighterTargetArea
import com.intellij.openapi.editor.markup.TextAttributes
import com.intellij.openapi.fileEditor.FileEditorManager
import com.intellij.openapi.project.DumbAware
import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.Messages
import com.intellij.openapi.util.Disposer
import com.intellij.openapi.vfs.VirtualFile
import com.intellij.psi.PsiDocumentManager
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

        // Clear previous demo highlighters if any
        DemoHighlighters.removeForEditor(editor)

        // Add colored highlights for the demo method ranges
        addDemoHighlights(editor)

        // Show a small dialog with the hardcoded report
        val psiFile = PsiManager.getInstance(project).findFile(vFile)
        val fileName = psiFile?.name ?: vFile.name
        CleanCodeAnalysisDialog(project, fileName) { openDocs() }.show()

        Notifications.Bus.notify(
            Notification(
                "AI Clean Code Analysis",
                "AI Clean Code Analysis Finished",
                "Demo results generated for ${vFile.name}",
                NotificationType.INFORMATION
            ),
            project
        )
    }

    private fun addDemoHighlights(editor: Editor) {
        val doc = editor.document
        fun clampLine(line: Int) = line.coerceIn(1, doc.lineCount) // 1-based input

        val yellow = TextAttributes(null, JBColor(Color(255, 243, 205), Color(58, 50, 35)), JBColor.YELLOW.darker(), EffectType.BOXED, Font.PLAIN)
        val red = TextAttributes(null, JBColor(Color(255, 235, 238), Color(63, 40, 43)), JBColor.RED, EffectType.BOXED, Font.BOLD)

        val recStart = doc.getLineStartOffset(clampLine(12) - 1)
        val recEnd = doc.getLineEndOffset(clampLine(15) - 1)
        val reqStart = doc.getLineStartOffset(clampLine(17) - 1)
        val reqEnd = doc.getLineEndOffset(clampLine(20) - 1)

        val markup = editor.markupModel
        val yellowHl = markup.addRangeHighlighter(recStart, recEnd, HighlighterLayer.SELECTION - 1, yellow, HighlighterTargetArea.EXACT_RANGE)
        val redHl = markup.addRangeHighlighter(reqStart, reqEnd, HighlighterLayer.SELECTION - 1, red, HighlighterTargetArea.EXACT_RANGE)

        // Add thin colored line markers in the editor gutter (error stripe)
        val errorStripeYellow = TextAttributes(null, null, JBColor(0xFFC107, 0xFFC107), EffectType.LINE_UNDERSCORE, Font.PLAIN)
        val errorStripeRed = TextAttributes(null, null, JBColor(0xE53935, 0xE53935), EffectType.LINE_UNDERSCORE, Font.BOLD)
        val stripeYellow = markup.addRangeHighlighter(recStart, recEnd, HighlighterLayer.WARNING, errorStripeYellow, HighlighterTargetArea.EXACT_RANGE)
        val stripeRed = markup.addRangeHighlighter(reqStart, reqEnd, HighlighterLayer.ERROR, errorStripeRed, HighlighterTargetArea.EXACT_RANGE)

        DemoHighlighters.storeForEditor(editor, listOf(yellowHl, redHl, stripeYellow, stripeRed))
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

    fun storeForEditor(editor: Editor, list: List<com.intellij.openapi.editor.markup.RangeHighlighter>) {
        removeForEditor(editor)
        editor.putUserData(EditorHighlighterKey, list)
    }

    private val EditorHighlighterKey = com.intellij.openapi.util.Key.create<List<com.intellij.openapi.editor.markup.RangeHighlighter>>(key)
}
