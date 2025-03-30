//
//  LocalAIChatScreenView.swift
//  LocalGenAIApp
//
//  Created by ikeuchi.ryuto on 2025/03/01.
//

import Foundation
import MLX
import MLXRandom
import MarkdownUI
import Metal
import SwiftUI
import Tokenizers

struct LocalAIChatScreenView: View {
    @Environment(\.dismiss) var dismiss
    @Environment(DeviceStat.self) private var deviceStat

    @State var slm = SLMEvaluator()
    @State var userPrompt = ""
    @State private var messages: [ChatMessage] = [
        ChatMessage(content: "こんにちは", role: "user"),
        ChatMessage(content: "こんにちは!Phi-4-miniです。", role: "ai"),
    ]
    @State private var cpuUsage: Float = 0.0
    @State private var memoryUsage: UInt64 = 0
    
    let timer = Timer.publish(every: 1.0, on: .main, in: .common).autoconnect()

    var body: some View {
        NavigationView {
            VStack(alignment: .leading) {
                Text(String(format: "CPU Using: %.2f%%", cpuUsage))
                Text("Memory Using: \(memoryUsage / 1024 / 1024) MB")
                    .font(.system(size: 16))
                ScrollView(.vertical) {
                    ScrollViewReader { sp in
                        Group {
                            ForEach(messages) { message in
                                HStack {
                                    ChatBubble(message: message)
                                }
                            }
                        }
                        .onChange(of: slm.output) { newOutput in
                            if slm.running,
                                let lastIndex = messages.indices.last,
                                messages[lastIndex].role == "ai"
                            {
                                messages[lastIndex].content = newOutput
                            }
                            sp.scrollTo("bottom")
                        }

                        Spacer().frame(width: 1, height: 1).id("bottom")
                    }
                }

                HStack {
                    TextField("メッセージを入力", text: $userPrompt).font(.title3)
                        .bold().onSubmit(generate).disabled(
                            slm.running)
                    Button(action: generate) {
                        Image(systemName: "mic.fill").foregroundStyle(.blue)
                            .frame(width: 24, height: 24)
                    }.disabled(slm.running)
                }.padding(.leading, 40).padding(.trailing, 32).padding(
                    .vertical, 28
                ).background(
                    Color.white
                ).cornerRadius(40).overlay(
                    RoundedRectangle(cornerRadius: 40).stroke(.white))

            }
            .padding(.horizontal, 60)
            .background(.BACKGROUND)
            .navigationBarTitle(
                "MLX形式で検証"
            )
            .navigationBarTitleDisplayMode(.inline)
        }
        .navigationViewStyle(.stack)
        .task {
            updateMetrics()
            _ = try? await slm.load()
        }
        .onReceive(timer) { _ in
            updateMetrics()
        }
    }

    private func generate() {
        Task {
            messages.append(ChatMessage(content: userPrompt, role: "user"))
            messages.append(ChatMessage(content: "", role: "ai"))
            await slm.generate(prompt: userPrompt)
            userPrompt = ""
        }
    }
    
    func updateMetrics() {
        cpuUsage = getCPUUsage()
        memoryUsage = getMemoryUsage()
    }

    func getCPUUsage() -> Float {
        var kr: kern_return_t

        var threadList: thread_act_array_t?
        var threadCount = mach_msg_type_number_t(0)
        kr = task_threads(mach_task_self_, &threadList, &threadCount)
        if kr != KERN_SUCCESS {
            return -1
        }

        var totalUsage: Float = 0.0
        if let threadList = threadList {
            for i in 0..<Int(threadCount) {
                var threadInfo = thread_basic_info()
                print(threadInfo)
                var threadInfoCount = mach_msg_type_number_t(THREAD_INFO_MAX)
                kr = withUnsafeMutablePointer(to: &threadInfo) {
                    threadInfoPtr in
                    threadInfoPtr.withMemoryRebound(
                        to: integer_t.self, capacity: Int(threadInfoCount)
                    ) { intPtr in
                        thread_info(
                            threadList[i],
                            thread_flavor_t(THREAD_BASIC_INFO),
                            intPtr,
                            &threadInfoCount)
                    }
                }
                if kr != KERN_SUCCESS {
                    continue
                }
                if (threadInfo.flags & TH_FLAGS_IDLE) == 0 {
                    totalUsage +=
                        Float(threadInfo.cpu_usage) / Float(TH_USAGE_SCALE)
                        * 100.0
                }
            }
        }

        // 取得したスレッドリストのメモリを解放
        if let threadList = threadList {
            let size =
                vm_size_t(threadCount) * vm_size_t(MemoryLayout<thread_t>.size)
            vm_deallocate(
                mach_task_self_, vm_address_t(bitPattern: threadList), size)
        }
        return totalUsage
    }

    func getMemoryUsage() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / 4)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                pointer in
                task_info(
                    mach_task_self_, task_flavor_t(TASK_VM_INFO), pointer,
                    &count)
            }
        }

        if kr != KERN_SUCCESS {
            return 0
        }
        // info.phys_footprint は実際に使用されている物理メモリ量（バイト単位）
        return info.phys_footprint
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
                        .background(
                            UnevenRoundedRectangle(
                                topLeadingRadius: 16,
                                bottomLeadingRadius: 16,
                                bottomTrailingRadius: 0,
                                topTrailingRadius: 16,
                                style: .continuous
                            ).foregroundStyle(.white)
                        )
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
                    HStack {
                        Image("Phi4MiniIcon").resizable().scaledToFit().frame(
                            width: 50, height: 50
                        ).clipShape(
                            Circle())
                        Markdown(message.content)
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
                            .padding(16)
                            .background(
                                UnevenRoundedRectangle(
                                    topLeadingRadius: 16,
                                    bottomLeadingRadius: 0,
                                    bottomTrailingRadius: 16,
                                    topTrailingRadius: 16,
                                    style: .continuous
                                ).foregroundStyle(.white)
                            )
                    }
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
