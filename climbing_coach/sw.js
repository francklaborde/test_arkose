// Minimal service worker — required for "Add to Home Screen" to count as an installable PWA.
// No offline caching yet; just passes requests through.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", () => self.clients.claim());
self.addEventListener("fetch", () => {});
