// Marque les aimants cochés (✔️) et le bouton qui fait avancer (✅), pour leur donner un style.
(function () {
  function mark() {
    document.querySelectorAll(".ai-message .flex-wrap > button").forEach(function (button) {
      var text = (button.innerText || "").trim();
      button.classList.toggle("is-on", text.indexOf("✔") === 0);
      button.classList.toggle("is-go", text.indexOf("✅") === 0);
    });
  }
  new MutationObserver(mark).observe(document.documentElement, { childList: true, subtree: true });
})();
