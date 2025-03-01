//
//  Item.swift
//  LocalGenAIApp
//
//  Created by ikeuchi.ryuto on 2025/03/01.
//

import Foundation
import SwiftData

@Model
final class Item {
    var timestamp: Date
    
    init(timestamp: Date) {
        self.timestamp = timestamp
    }
}
