//
//  LocalGenAIAppApp.swift
//  LocalGenAIApp
//
//  Created by ikeuchi.ryuto on 2025/03/01.
//

import SwiftData
import SwiftUI

@main
struct LocalGenAIAppApp: App {
  var sharedModelContainer: ModelContainer = {
    let schema = Schema([
      Item.self
    ])
    let modelConfiguration = ModelConfiguration(schema: schema, isStoredInMemoryOnly: false)

    do {
      return try ModelContainer(for: schema, configurations: [modelConfiguration])
    } catch {
      fatalError("Could not create ModelContainer: \(error)")
    }
  }()

  var body: some Scene {
    WindowGroup {
      LocalAIChatScreenView().environment(DeviceStat())
    }
    .modelContainer(sharedModelContainer)
  }
}
