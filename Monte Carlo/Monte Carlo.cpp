#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <Eigen/Dense>

using namespace Eigen;

// Function to simulate futures price paths using Geometric Brownian Motion
std::vector<double> simulate_futures_path(double F0, double r, double sigma, double T, int steps, std::mt19937& rng) {
	std::normal_distribution<double> dist(0, 1);// Standard normal distribution for random sampling
	double dt = T / steps;						// Time step size
	std::vector<double> path(steps + 1, F0);	// Vector to store the simulated path, initialized with F0

	for (int i = 1; i <= steps; ++i) {
		double dW = dist(rng) * sqrt(dt);		// Increment of Brownian motion			
		path[i] = path[i - 1] * exp((r - 0.5 * sigma * sigma) * dt + sigma + dW); // Risk-neutral dynamics
	}

	return path;
}

// Function to calculate continuation values using Least-Squares Regression
std::vector<double> calculate_continuation_values(const std::vector<std::vector<double>>& paths, const std::vector<double>& payoffs, int step) {
	std::vector<double> continuation_values(payoffs.size(), 0.0);	// Initialize continuation values with 0
	std::vector<double> X, Y;	// Vectors to store independent and dependent variables for regression

	// Collect in-the-money paths and payoffs
	for (size_t i = 0; i < paths.size(); ++i) {
		if (paths[i][step] > 0) { // In-the-money check
			X.push_back(paths[i][step]);
			Y.push_back(payoffs[i]);
		}
	}

	// If no in-the-money paths, return default continuation values
	if (X.empty()) return continuation_values;

	// Perform least-squares regression (using Eigen)
	MatrixXd X_matrix(X.size(), 2);	// Matrix for regression inputs (1 for constant, 1 for futures price)
	VectorXd Y_vector(Y.size());	// Vector for regression outputs (future payoffs)

	for (size_t i = 0; i < X.size(); ++i) {
		X_matrix(i, 0) = 1;           // Constant term
		X_matrix(i, 1) = X[i];        // Futures price
		Y_vector(i) = Y[i];           // Corresponding payoff
	}

	// Solve for regression coefficients
	VectorXd beta = X_matrix.jacobiSvd(ComputeThinU | ComputeThinV).solve(Y_vector);

	// Compute continuation values for all paths
	for (size_t i = 0; i < paths.size(); ++i) {
		if (paths[i][step] > 0) { // Only compute for in-the-money paths
			continuation_values[i] = beta[0] + beta[1] * paths[i][step];
		}
	}

	return continuation_values;

}


// Monte Carlo simulation with early exercise
double price_american_futures_option(double F0, double K, double T, double r, double sigma, int steps, int simulations) {
	std::random_device rd;		// Random device for seed
	std::mt19937 rng(rd());		// Mersenne Twister RNG for reproducibility

	// Generate paths
	std::vector<std::vector<double>> paths(simulations); // Vector of paths
	for (int i = 0; i < simulations; ++i) {
		paths[i] = simulate_futures_path(F0, r, sigma, T, steps, rng); 
	}

	// Initialize payoffs at maturity
	std::vector<double> payoffs(simulations, 0.0); // Vector to store payoffs
	for (int i = 0; i < simulations; ++i) {
		payoffs[i] = std::max(paths[i][steps] - K, 0.0); // Payoff for a call option
	}

	// Backward induction
	for (int step = steps - 1; step >= 0; --step) {
		// Compute continuation values using regression
		std::vector<double> continuation_values = calculate_continuation_values(paths, payoffs, step);
		
		// Update payoffs by considering early exercise
		for (int i = 0; i < simulations; ++i) {
			double instrinsic_value = std::max(paths[i][step] - K, 0.0); // Value if exercised early
			payoffs[i] = std::max(instrinsic_value, continuation_values[i]); // Optimal value at this step
		}
	}

	// Discounted average payoff
	double option_price = 0.0;
	for (double payoff : payoffs) {
		option_price += payoff; // Sum up all payoffs
	}
	option_price = exp(-r * T) * (option_price / simulations); // Discount to present value

	return option_price;
}

int main() {
	double F0 = 100.0; // Futures price
	double K = 100.0;  // Strike price
	double T = 1.0;    // Time to maturity (years)
	double r = 0.05;   // Risk-free rate
	double sigma = 0.2;// Volatility
	int steps = 100;   // Time steps
	int simulations = 10000; // Number of simulations

	double option_price = price_american_futures_option(F0, K, T, r, sigma, steps, simulations);

	std::cout << "American Call Option Price on Futures: " << option_price << std::endl;

	return 0;
}