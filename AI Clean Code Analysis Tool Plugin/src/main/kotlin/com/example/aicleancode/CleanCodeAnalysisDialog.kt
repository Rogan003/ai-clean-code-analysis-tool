package com.example.aicleancode

import com.intellij.openapi.project.Project
import com.intellij.openapi.ui.DialogPanel
import com.intellij.openapi.ui.DialogWrapper
import com.intellij.ui.JBColor
import com.intellij.ui.components.JBLabel
import com.intellij.ui.dsl.builder.Align
import com.intellij.ui.dsl.builder.COLUMNS_LARGE
import com.intellij.ui.dsl.builder.panel
import java.awt.Color
import javax.swing.Icon
import javax.swing.JComponent
import javax.swing.UIManager

class CleanCodeAnalysisDialog(
    project: Project,
    private val fileName: String,
    private val onLearnMore: () -> Unit
) : DialogWrapper(project, true) {

    init {
        title = "AI Clean Code Analysis"
        init()
    }

    override fun createCenterPanel(): JComponent {
        val successColor = JBColor(Color(56, 142, 60), Color(129, 199, 132))
        val warningColor = JBColor(Color(255, 193, 7), Color(255, 213, 79))
        val errorColor = JBColor(Color(229, 57, 53), Color(239, 154, 154))

        val panel: DialogPanel = panel {
            row { label("Report for: $fileName").bold() }
            row { 
                cell(statusPill("Good: Class follows clean code guidelines", successColor))
                    .align(Align.FILL)
            }
            row {
                cell(statusPill("Changes recommended", warningColor))
                    .align(Align.FILL)
            }
            row {
                cell(statusPill("Changes required", errorColor))
                    .align(Align.FILL)
            }
            separator()
            row {
                link("Learn more about clean code analysis docs") { onLearnMore() }
                comment("This is a demo with hardcoded results. Model integration will come later.")
            }
        }
        return panel
    }

    private fun statusPill(text: String, bg: JBColor): JComponent {
        val label = JBLabel(text)
        label.isOpaque = true
        label.foreground = JBColor(Color(33, 33, 33), Color(250, 250, 250))
        label.background = bg
        label.border = javax.swing.BorderFactory.createEmptyBorder(8, 10, 8, 10)
        return label
    }
}
