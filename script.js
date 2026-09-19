document.addEventListener("DOMContentLoaded", function () {

  // Scroll Offsets
  const DESKTOP_SCROLL_OFFSET = 60;
  const MOBILE_SCROLL_OFFSET = 30;
  const MOBILE_BREAKPOINT = 768;

  // Typing Animation
  const typingEl = document.querySelector('.typing');
  if (typingEl) {
    const text = typingEl.textContent;
    typingEl.textContent = '';
    let i = 0;

    function type() {
      if (i < text.length) {
        typingEl.textContent += text.charAt(i++);
        setTimeout(type, 100);
      } else {
        // Add a class when typing is done to hide the blinking cursor via CSS
        typingEl.classList.add('typing-finished');
      }
    }
    type();
  }

  // Dark Mode Toggle
  const toggleCheckbox = document.getElementById("toggle-mode");
  if (toggleCheckbox) {
    // Load preference from localStorage
    if (localStorage.getItem("darkMode") === "true") {
      document.body.classList.add("dark-mode");
      toggleCheckbox.checked = true;
    }

    toggleCheckbox.addEventListener("change", () => {
      document.body.classList.toggle("dark-mode");
      // Save preference to localStorage
      const isDark = document.body.classList.contains("dark-mode");
      localStorage.setItem("darkMode", isDark);
    });
  }

  // Elements used by multiple handlers
  const hamburger = document.getElementById('hamburger');
  const navUl = document.getElementById('nav-ul');

  // Hamburger Menu
  if (hamburger && navUl) {
    hamburger.addEventListener('click', () => {
      const isActive = navUl.classList.toggle('active');
      hamburger.setAttribute('aria-expanded', isActive);
      hamburger.innerHTML = isActive
        ? '<i class="fas fa-times"></i>'
        : '<i class="fas fa-bars"></i>';
    });

    // Close menu when a link is clicked (keeps previous behavior)
    document.querySelectorAll('#nav-ul a').forEach(link => {
      link.addEventListener('click', () => {
        navUl.classList.remove('active');
        hamburger.setAttribute('aria-expanded', 'false');
        hamburger.innerHTML = '<i class="fas fa-bars"></i>';
      });
    });
  }

  // ScrollSpy
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".navbar a[href^='#']");
  // Compute offsets inside the scroll listener so they respond to viewport width changes
  if (sections.length > 0 && navLinks.length > 0) {
    window.addEventListener("scroll", () => {
      let current = "";
      const offsetForSpy = window.innerWidth >= MOBILE_BREAKPOINT ? DESKTOP_SCROLL_OFFSET : MOBILE_SCROLL_OFFSET;

      sections.forEach(section => {
        const sectionTop = section.offsetTop - offsetForSpy - 10; // small buffer
        if (window.scrollY >= sectionTop) {
          current = section.getAttribute("id");
        }
      });

      navLinks.forEach(link => {
        link.classList.remove("active");
        // match href exactly "#id" or contains (safer)
        const href = link.getAttribute("href") || "";
        if (current && (href === `#${current}` || href.includes(`#${current}`))) {
          link.classList.add("active");
        }
      });
    });
  }

  // Smooth Scrolling with Offset
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      const href = this.getAttribute('href');
      // ignore empty or trivial hashes
      if (!href || href === '#') return;

      const targetEl = document.querySelector(href);
      if (!targetEl) return; // anchor points to nothing in page

      // Prevent default jump
      e.preventDefault();

      const offset = window.innerWidth >= MOBILE_BREAKPOINT ? DESKTOP_SCROLL_OFFSET : MOBILE_SCROLL_OFFSET;
      const targetPosition = Math.max(0, targetEl.getBoundingClientRect().top + window.scrollY - offset);

      window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
      });

      // If mobile nav is open, close it after clicking
      if (navUl && navUl.classList.contains('active')) {
        navUl.classList.remove('active');
        if (hamburger) {
          hamburger.setAttribute('aria-expanded', 'false');
          hamburger.innerHTML = '<i class="fas fa-bars"></i>';
        }
      }
    });
  });

  // Contact Form with Formspree
  const form = document.getElementById("contact-form");
  const popup = document.getElementById("thank-you-popup");
  if (form && popup) {
    form.addEventListener("submit", function (e) {
      e.preventDefault();
      const formData = new FormData(form);

      fetch("https://formspree.io/f/xrpzokla", {
        method: "POST",
        body: formData,
        headers: {
          'Accept': 'application/json'
        }
      })
      .then(response => {
        if (response.ok) {
          form.reset();
          popup.classList.add("show");
          setTimeout(() => {
            popup.classList.remove("show");
          }, 4000);
        } else {
          // Error message
          console.error("Oops! Something went wrong while submitting.");
          alert("Sorry, there was an error sending your message. Please try again later.");
        }
      })
      .catch(() => {
        console.error("Error sending form. Please check your connection.");
        alert("Sorry, there was a network error. Please check your connection and try again.");
      });
    });
  }
    
  // Dynamic Copyright Year
  const yearSpan = document.getElementById("current-year");
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear();
  }
});
