import Darwin
import ServiceManagement
import SwiftUI

@main
enum QWMenuBarMain {
    @MainActor
    static func main() async {
        if CommandLine.arguments.contains("--unregister-login-item") {
            do {
                if SMAppService.mainApp.status == .enabled || SMAppService.mainApp.status == .requiresApproval {
                    try await SMAppService.mainApp.unregister()
                }
                print("Qwegn login item unregistered.")
            } catch {
                fputs("Qwegn: could not unregister login item: \(error.localizedDescription)\n", stderr)
                exit(1)
            }
            return
        }
        QWMenuBarApp.main()
    }
}
