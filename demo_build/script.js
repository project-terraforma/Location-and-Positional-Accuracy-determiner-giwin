// Scrollytelling Implementation
document.addEventListener('DOMContentLoaded', () => {
    // Select all the narrative steps and map wrappers
    const steps = document.querySelectorAll('.step');
    const maps = document.querySelectorAll('.map-wrapper');

    // Create an intersection observer to watch which step is active
    const observerOptions = {
        root: null, // use viewport
        rootMargin: '-30% 0px -70% 0px', // Trigger when step crosses the top 30% mark of the screen
        threshold: 0
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Get the step number from data-attribute
                const stepNum = entry.target.getAttribute('data-step');

                // Update active state of steps
                steps.forEach(step => step.classList.remove('is-active'));
                entry.target.classList.add('is-active');

                // Update active map
                maps.forEach(map => map.classList.remove('active'));
                const activeMap = document.getElementById(`map-${stepNum}`);
                if (activeMap) {
                    activeMap.classList.add('active');
                }
            }
        });
    }, observerOptions);

    // Observer starts observing all steps
    steps.forEach(step => observer.observe(step));

    // Ensure the first step is active on load
    if (steps.length > 0) {
        steps[0].classList.add('is-active');
    }
});
