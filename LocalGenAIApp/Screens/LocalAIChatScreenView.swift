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

struct LocalAIChatScreenView: View {
    @Environment(\.dismiss) var dismiss
    
    @State var slm = SLMEvaluator()
    @State var userPrompt = ""
    
    enum displayStyle: String, CaseIterable, Identifiable {
        case plain, markdown
        var id: Self { self }
    }
    
    @State private var selectedDisplayStyle = displayStyle.markdown
    
    var body: some View {
        NavigationView {
            VStack(alignment: .leading) {
                VStack {
                    HStack {
                        Text(slm.modelInfo).textFieldStyle(.roundedBorder)
                        Spacer()
                        Text(slm.stat)
                    }
                    HStack {
                        Spacer()
                        if slm.running {
                            ProgressView()
                                .frame(maxHeight: 20)
                            Spacer()
                        }
                        Picker("", selection: $selectedDisplayStyle) {
                            ForEach(displayStyle.allCases, id: \.self) { option in
                                Text(option.rawValue.capitalized)
                                    .tag(option)
                            }
                            
                        }
                        .pickerStyle(.segmented).frame(maxWidth: 150)
                    }
                }
                
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            Text(slm.output).textSelection(.enabled)
                        }
                        .onChange(of: slm.output) { _, _ in
                            sp.scrollTo("bottom")
                        }
                        
                        Spacer().frame(width: 1, height: 1).id("bottom")
                    }
                }
                
                HStack {
                    TextField("prompt", text: $userPrompt).onSubmit(generate).disabled(slm.running)
                    Button("送信", action: generate).disabled(slm.running)
                }
            }.padding().toolbar {
                ToolbarItem {
                    Label("Memory Usage: OK", systemImage: "info.circle.fill").labelStyle(.titleAndIcon).padding(.horizontal)
                        .help(
                            Text(
                            """
                            Active Memory:ok /\(GPU.memoryLimit.formatted(.byteCount(style: .memory)))
                            Cache Memory: ok /\(GPU.cacheLimit.formatted(.byteCount(style: .memory)))
                            Peak Memory: ok
                            """
                            )
                        )
                }
            }
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
