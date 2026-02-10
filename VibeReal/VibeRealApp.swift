//
//  VibeRealApp.swift
//  VibeReal
//
//  Created by Tran Manh on 2026/02/10.
//

import SwiftUI
import CoreData

@main
struct VibeRealApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
