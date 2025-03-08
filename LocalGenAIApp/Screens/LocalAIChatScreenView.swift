//
//  LocalAIChatScreenView.swift
//  LocalGenAIApp
//
//  Created by ikeuchi.ryuto on 2025/03/01.
//

import SwiftUI
import MLX
import MLXRandom
import Metal
import Tokenizers
import MarkdownUI

struct LocalAIChatScreenView: View {
    @Environment(\.dismiss) var dismiss
    
    @State var slm = SLMEvaluator()
    @State var userPrompt = ""
    @State private var messages: [ChatMessage] = []
    
    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }
    
    @State private var selectedDisplayStyle = displayStyle.markdown
    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading) {
//                VStack {
//                    HStack {
//                        Text(slm.modelInfo).textFieldStyle(.roundedBorder)
//                        Spacer()
//                        Text(slm.stat)
//                    }
//                    HStack {
//                        Spacer()
//                        if slm.running {
//                            ProgressView()
//                                .frame(maxHeight: 20)
//                            Spacer()
//                        }
//                        Picker("", selection: $selectedDisplayStyle) {
//                            ForEach(displayStyle.allCases, id: \.self) { option in
//                                Text(option.rawValue.capitalized)
//                                    .tag(option)
//                            }
//                            
//                        }
//                        .pickerStyle(.segmented).frame(maxWidth: 150)
//                    }
//                }
                
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            List(messages) { message in
                                ChatBubble(message: message)
                            }
                            Markdown(slm.output)
                            .markdownBlockStyle(\.heading1) { configuration in
                                   configuration.label
                                    .padding(8)
                                    .markdownMargin(top: 0, bottom: 32)
                                    .markdownTextStyle {
                                        FontSize(24)
                                        FontWeight(.bold)
                                     }
                                 }
                            .markdownBlockStyle(\.heading2) { configuration in
                                   configuration.label
                                    .padding(8)
                                    .markdownMargin(top: 0, bottom: 16)
                                    .markdownTextStyle {
                                        FontSize(16)
                                        FontWeight(.bold)
                                     }
                                 }
                            .markdownBlockStyle(\.heading3) { configuration in
                                   configuration.label
                                    .markdownMargin(top: 16, bottom: 8)
                                    .markdownTextStyle {
                                        FontSize(18)
                                        FontWeight(.bold)
                                     }
                                 }
                                .textSelection(.enabled)
                                .frame(width:  UIScreen.main.bounds.width - 80, alignment: .leading)
                        }
                        .onChange(of: slm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }
                        
                        Spacer().frame(width: 1, height: 1).id("bottom")
                    }
                }
                
                HStack {
                    TextField("メッセージを入力", text: $userPrompt).font(.title3).bold().onSubmit(generate).disabled(slm.running)
                    Button(action: generate) {
                        Image(systemName: "mic.fill").foregroundStyle(.blue).frame(width: 24, height: 24)
                    }.disabled(slm.running)
                }.padding(.leading, 40).padding(.trailing, 32).padding(.vertical, 28).background(Color.white).cornerRadius(40).overlay(RoundedRectangle(cornerRadius: 40).stroke(.white))
                
            }
            .padding().toolbar {
//                ToolbarItem {
//                    Label("Memory Usage: OK", systemImage: "info.circle.fill").labelStyle(.titleAndIcon).padding(.horizontal)
//                        .help(
//                            Text(
//                            """
//                            Active Memory:ok /\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
//                            Cache Memory: ok /\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
//                            Peak Memory: ok
//                            """
//                            )
//                        )
//                }
            }.background(.BACKGROUND)
        }.navigationViewStyle(.stack)
    }
    
    private func generate() {
        Task {
            await slm.generate(prompt: userPrompt)
        }
    }
}

#Preview {
    LocalAIChatScreenView()
}

struct ChatMessage: Identifiable {
    var id = UUID()
    var content: String
    var role: String
}

struct ChatBubble: View {
    var message: ChatMessage

    var body: some View {
        Group {
            if message.role == "user" {
                HStack {
                    Spacer()
                    VStack(alignment: .trailing, spacing: 2) {
                        Text(message.content)
                            .padding(12)
                            .background(Color.blue)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                        Text("Sent: \(getMessageSentTime())")
                            .font(.system(size: 8, weight: .medium))
                    }
                }
            } else {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text("John")
                            .font(.system(size: 12, weight: .medium))
                        Text(message.content)
                            .padding(12)
                            .background(Color.gray)
                            .foregroundColor(.white)
                            .cornerRadius(10)
                    }
                    Spacer()
                }
            }
        }
    }

    private func getMessageSentTime() -> String {
        let date = NSDate()
        let formatter = DateFormatter()
        formatter.dateFormat = "HH:mm"
        return formatter.string(from: date as Date)
    }

}
