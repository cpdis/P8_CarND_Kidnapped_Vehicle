/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

  if (is_initialized) {
    return;
  }

  // Initialize the number of particles. 
  num_particles = 100;

  // Creating the normal distributions.
  normal_distribution<double> dist_x(x, std[0]); // x standard deviation
  normal_distribution<double> dist_y(y, std[1]); // y standard deviation
  normal_distribution<double> dist_theta(theta, std[2]); // theta standard deviation

  // Create a normal distribution of particles with mean centered on GPS values.
  for (int i = 0; i < num_particles; i++) {

    Particle particle;
    particle.id = i;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;

    particles.push_back(particle);
	}

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Creating normal distributions
  normal_distribution<double> dist_x(0, std_pos[0]); // x standard deviation
  normal_distribution<double> dist_y(0, std_pos[1]); // y standard deviation
  normal_distribution<double> dist_theta(0, std_pos[2]); // theta standard deviation

  // Calculate new state.
  for (int i = 0; i < num_particles; i++) {

  	double theta = particles[i].theta;

    if ( fabs(yaw_rate) < 0.00001 ) {
      particles[i].x += velocity * delta_t * cos( theta );
      particles[i].y += velocity * delta_t * sin( theta );
    } else {
      particles[i].x += velocity / yaw_rate * ( sin( theta + yaw_rate * delta_t ) - sin( theta ) );
      particles[i].y += velocity / yaw_rate * ( cos( theta ) - cos( theta + yaw_rate * delta_t ) );
      particles[i].theta += yaw_rate * delta_t;
    }

    // Noise.
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

  unsigned int nObservations = observations.size();
  unsigned int nPredictions = predicted.size();

  for (unsigned int i = 0; i < nObservations; i++) {

    // Initialize the minimum distance with maximum possible distance.
    double minDistance = numeric_limits<double>::max();

    // Initial id of landmark
    int mapId = -1;

    for (unsigned j = 0; j < nPredictions; j++ ) {

      double xDistance = observations[i].x - predicted[j].x;
      double yDistance = observations[i].y - predicted[j].y;

      // Calculate distance between current and predicted landmarks.
      double distance = xDistance * xDistance + yDistance * yDistance;

      // Find the predicted landmark that is nearest the current observed landmark.
      if ( distance < minDistance ) {
        minDistance = distance;
        mapId = predicted[j].id;
      }
    }

    // Set the observation id to the nearest predicted landmark id.
    observations[i].id = mapId;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
  
  double stdRange = std_landmark[0];
  double stdBearing = std_landmark[1];

  // For each particle...
  for (int i = 0; i < num_particles; i++) {

    double px = particles[i].x;
    double py = particles[i].y;
    double theta = particles[i].theta;

    // Create a vector to hold the landmark locations that are predicted to be within range of the particle.
    vector<LandmarkObs> inRangeLandmarks;

    double sensor_range_2 = sensor_range * sensor_range;

    for(unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
      // Get id and x and y coordinates
      int id = map_landmarks.landmark_list[j].id_i;
      float landmarkX = map_landmarks.landmark_list[j].x_f;
      float landmarkY = map_landmarks.landmark_list[j].y_f;

      double dX = px - landmarkX;
      double dY = py - landmarkY;

      // Only consider landmarks within sensor range of the particle.
      if ( dX*dX + dY*dY <= sensor_range_2 ) {
        // Add the prediction to the inRange vector
        inRangeLandmarks.push_back(LandmarkObs{ id, landmarkX, landmarkY });
      }
    }

    // Create a list of observations transformed from the vehicle coordinates to map coordinates.
    vector<LandmarkObs> mappedObservations;

    for(unsigned int j = 0; j < observations.size(); j++) {
      int observation_id = observations[j].id;
      double observation_x = observations[j].x;
      double observation_y = observations[j].y;

      double transformed_x = cos(theta)*observation_x - sin(theta)*observation_y + px;
      double transformed_y = sin(theta)*observation_x + cos(theta)*observation_y + py;

      mappedObservations.push_back(LandmarkObs{observation_id, transformed_x, transformed_y});
    }

    // Use dataAssociation for the predictions and transformed observations
    dataAssociation(inRangeLandmarks, mappedObservations);

    // Reinitialize weight.
    particles[i].weight = 1.0;
    
    // Calculate weights.
    for(unsigned int j = 0; j < mappedObservations.size(); j++) {
      // Observation and the associated prediction coordinates.
      int landmarkId = mappedObservations[j].id;
      double observationX = mappedObservations[j].x;
      double observationY = mappedObservations[j].y;

      double landmarkX, landmarkY;
      
      unsigned int k = 0;
      unsigned int nLandmarks = inRangeLandmarks.size();
      
      bool found = false;
      
      while( !found && k < nLandmarks ) {
        if ( inRangeLandmarks[k].id == landmarkId) {
          found = true;
          landmarkX = inRangeLandmarks[k].x;
          landmarkY = inRangeLandmarks[k].y;
        }
        k++;
      }

      // Calculating weight.
      double dX = observationX - landmarkX;
      double dY = observationY - landmarkY;

      double weight = ( 1/(2*M_PI*stdRange*stdBearing)) * exp( -( dX*dX/(2*stdRange*stdRange) + (dY*dY/(2*stdBearing*stdBearing)) ) );
      
      if (weight == 0) {
        particles[i].weight *= 0.00001;
      } else {
        particles[i].weight *= weight;
      }
    }
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  // Get weights and maximum weight.
  vector<double> weights;
  double maximumWeight = numeric_limits<double>::min();

  for(int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
    
    if ( particles[i].weight > maximumWeight ) {
      maximumWeight = particles[i].weight;
    }
  }

  // Creating distributions.
  uniform_real_distribution<double> distDouble(0.0, maximumWeight);
  uniform_int_distribution<int> distInt(0, num_particles - 1);

  // Generate index.
  int index = distInt(gen);

  double beta = 0.0;

  // Resample wheel.
  vector<Particle> resampledParticles;

  for(int i = 0; i < num_particles; i++) {
    beta += distDouble(gen) * 2.0;
    
    while( beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampledParticles.push_back(particles[index]);
  }

  particles = resampledParticles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}