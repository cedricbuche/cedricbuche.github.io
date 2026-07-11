document.addEventListener('DOMContentLoaded', function () {

  // Load sidebar
  fetch("sidebar.html")
    .then(response => {
      if (!response.ok) {
        throw new Error("Cannot load sidebar.html");
      }
      return response.text();
    })
    .then(data => {

      var sidebar = document.getElementById("sidebar");

      if (sidebar) {
        sidebar.innerHTML = data;
      }

      // Mobile nav toggle (after sidebar is loaded)
      var toggle = document.querySelector('.nav-toggle');
      var nav = document.querySelector('.nav');

      if (toggle && nav) {
        toggle.addEventListener('click', function () {
          nav.classList.toggle('open');
          var expanded = nav.classList.contains('open');
          toggle.setAttribute('aria-expanded', expanded);
          toggle.textContent = expanded ? 'Close ×' : 'Menu ☰';
        });
      }

      // Highlight current page
      var here = location.pathname.split('/').pop() || 'index.html';

      document.querySelectorAll('.nav a').forEach(function (a) {
        var href = a.getAttribute('href');

        if (href === here) {
          a.classList.add('active');
        }
      });

    })
    .catch(error => {
      console.error("Sidebar error:", error);
    });


  // Publication filters
  var filterBtns = document.querySelectorAll('.pub-filters button');

  if (filterBtns.length) {

    filterBtns.forEach(function (btn) {

      btn.addEventListener('click', function () {

        filterBtns.forEach(function (b) {
          b.classList.remove('active');
        });

        btn.classList.add('active');

        var cat = btn.dataset.cat;

        document.querySelectorAll('.pub-section').forEach(function (sec) {
          sec.style.display =
            (cat === 'all' || sec.dataset.cat === cat)
            ? ''
            : 'none';
        });

      });

    });

  }

});
