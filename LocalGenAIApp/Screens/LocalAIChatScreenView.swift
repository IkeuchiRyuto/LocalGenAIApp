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
import Foundation

struct LocalAIChatScreenView: View {
    @Environment(\.dismiss) var dismiss
        
    @State var slm = SLMEvaluator()
    @State var userPrompt = ""
    @State private var messages: [ChatMessage] = [
        ChatMessage(content: "こんにちは", role: "user"),
        ChatMessage(content: "こんにちは!!何を答えましょうか？", role: "ai")
    ]

    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading) {
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            ForEach(messages) { message in
                                HStack {
                                    ChatBubble(message: message)
                                }
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
            .padding(.horizontal, 60)
            .background(.BACKGROUND)
            .navigationBarTitle("\(slm.modelInfo) GPU：\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))")
            .navigationBarTitleDisplayMode(.inline)
        }.navigationViewStyle(.stack)
    }
    
    private func generate() {
//        Task {
            messages.append(ChatMessage(content: "こんにちは", role: "user"))
            userPrompt = ""
//            await slm.generate(prompt: "こんにちは")
//        }
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
        if message.role == "user" {
            HStack {
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text(message.content)
                        .padding(16)
                        .background(UnevenRoundedRectangle(
                            topLeadingRadius: 16,
                            bottomLeadingRadius: 16,
                            bottomTrailingRadius: 0,
                            topTrailingRadius: 16,
                            style: .continuous
                        ).foregroundStyle(.white))
                        .foregroundColor(.blue)
                        .fontWeight(.bold)
                    Text("Sent: \(getMessageSentTime())")
                        .font(.system(size: 8, weight: .medium))
                }
            }
        } else {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Phi-4-mini")
                        .font(.system(size: 12, weight: .medium))
                        .padding(.bottom, 4)
                    Text(message.content)
                        .padding(16)
                        .background(UnevenRoundedRectangle(
                            topLeadingRadius: 16,
                            bottomLeadingRadius: 0,
                            bottomTrailingRadius: 16,
                            topTrailingRadius: 16,
                            style: .continuous
                        ).foregroundStyle(.white))
                        .foregroundColor(.black)
                }
                Spacer()
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
