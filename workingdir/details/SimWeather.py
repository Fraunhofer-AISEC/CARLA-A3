import carla

class Weather:
    def __init__(
        self,
        cloudiness=30.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        sun_azimuth_angle=100.0,
        sun_altitude_angle=70.0,
        fog_density=0.0, 
        fog_distance=0.0, 
        wetness=0.0, 
        fog_falloff=0.2, 
        scattering_intensity=0.0, 
        mie_scattering_scale=0.0, 
        rayleigh_scattering_scale=0.0331
    ) -> None:
        # Default weather
        self.cloudiness = cloudiness
        self.precipitation = precipitation
        self.precipitation_deposits = precipitation_deposits
        self.wind_intensity = wind_intensity
        self.sun_azimuth_angle = sun_azimuth_angle
        self.sun_altitude_angle = sun_altitude_angle
        self.fog_density = fog_density
        self.fog_distance = fog_distance
        self.wetness = wetness
        self.fog_falloff = fog_falloff
        self.scattering_intensity = scattering_intensity
        self.mie_scattering_scale = mie_scattering_scale
        self.rayleigh_scattering_scale = rayleigh_scattering_scale
        
        
    def toCarlaWeather(self):
        return carla.WeatherParameters(
            cloudiness = self.cloudiness,
            precipitation = self.precipitation, 
            precipitation_deposits = self.precipitation_deposits, 
            wind_intensity = self.wind_intensity, 
            sun_azimuth_angle = self.sun_azimuth_angle, 
            sun_altitude_angle = self.sun_altitude_angle, 
            fog_density = self.fog_density, 
            fog_distance = self.fog_distance, 
            wetness = self.wetness, 
            fog_falloff = self.fog_falloff, 
            scattering_intensity = self.scattering_intensity, 
            mie_scattering_scale = self.mie_scattering_scale, 
            rayleigh_scattering_scale = self.rayleigh_scattering_scale
        )


class PresetWeather:

    ClearNight = Weather(5.0000, 0.0000, 0.0000, 10.0000, -1.0000, -90.0000, 60.0000, 75.0000, 0.0000, 1.0000, 1.0000, 0.0300, 0.0331)
    ClearNoon = Weather(5.0000, 0.0000, 0.0000, 10.0000, -1.0000, 45.0000, 2.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    ClearSunset = Weather(5.0000, 0.0000, 0.0000, 10.0000, -1.0000, 15.0000, 2.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)

    CloudyNight = Weather(60.0000, 0.0000, 0.0000, 10.0000, -1.0000, -90.0000, 60.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    CloudyNoon = Weather(60.0000, 0.0000, 0.0000, 10.0000, -1.0000, 45.0000, 3.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    CloudySunset = Weather(60.0000, 0.0000, 0.0000, 10.0000, -1.0000, 15.0000, 3.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)

    HardRainyNight = Weather(100.0000, 100.0000, 90.0000, 100.0000, -1.0000, -90.0000, 100.0000, 0.7500, 100.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    HardRainyNoon = Weather(100.0000, 100.0000, 90.0000, 100.0000, -1.0000, 45.0000, 7.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    HardRainySunset = Weather(100.0000, 100.0000, 90.0000, 100.0000, -1.0000, 15.0000, 7.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)

    MidRainyNight = Weather(60.0000, 60.0000, 60.0000, 60.0000, -1.0000, 15.0000, 3.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    MidRainyNoon = Weather(80.0000, 60.0000, 60.0000, 60.0000, -1.0000, -90.0000, 60.0000, 0.7500, 80.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    MidRainySunset = Weather(60.0000, 60.0000, 60.0000, 60.0000, -1.0000, 45.0000, 3.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)

    SoftRainyNight = Weather(60.0000, 30.0000, 50.0000, 30.0000, -1.0000, -90.0000, 60.0000, 0.7500, 60.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    SoftRainyNoon = Weather(20.0000, 30.0000, 50.0000, 30.0000, -1.0000, 45.0000, 3.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    SoftRainySunset = Weather(20.0000, 30.0000, 50.0000, 30.0000, -1.0000, 15.0000, 2.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)

    WetCloudyNight = Weather(60.0000, 0.0000, 50.0000, 10.0000, -1.0000, -90.0000, 60.0000, 0.7500, 60.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    WetCloudyNoon = Weather(60.0000, 0.0000, 50.0000, 10.0000, -1.0000, 45.0000, 3.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    WetCloudySunset = Weather(60.0000, 0.0000, 50.0000, 10.0000, -1.0000, 15.0000, 2.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)

    WetNight = Weather(5.0000, 0.0000, 50.0000, 10.0000, -1.0000, -90.0000, 60.0000, 75.0000, 60.0000, 1.0000, 1.0000, 0.0300, 0.0331)
    WetNoon = Weather(5.0000, 0.0000, 50.0000, 10.0000, -1.0000, 45.0000, 3.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)
    WetSunset = Weather(5.0000, 0.0000, 50.0000, 10.0000, -1.0000, 15.0000, 2.0000, 0.7500, 0.0000, 0.1000, 1.0000, 0.0300, 0.0331)


        