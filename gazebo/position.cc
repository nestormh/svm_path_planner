#include <gazebo/gazebo.h>
#include <gazebo/GazeboError.hh>

int main()
{
	gazebo::Client *client = new gazebo::Client();
	gazebo::SimulationIface *simIface = new gazebo::SimulationIface();
	gazebo::PositionIface *posIface = new gazebo::PositionIface();

	int serverId = 0;

	/// Connect to the libgazebo server
	std::cout << "Tratamos de conectar con servidor\n" ;
	try
	{
		client->ConnectWait(serverId, GZ_CLIENT_ID_USER_FIRST);
	}
	catch (std::string e)
	{
		std::cout << "Gazebo error: Unable to connect\n" << e << "\n";
		return -1;
	}

	/// Open the Simulation Interface
	std::cout << "Tratamos de abrir simulacion\n" ;
	try
	{
		simIface->Open(client, "default");
	}
	catch (std::string e)
	{
		std::cout << "Gazebo error: Unable to connect to the sim interface\n" << e << "\n";
		return -1;
	}

	/// Open the Position interface
	std::cout << "Open the Position interface\n" ;
	try
	{
	posIface->Open(client, "simpleCar_model::position_iface_0");
//		 posIface->Open(client, "pioneer2dx_model1::position_iface_0");
	}
	catch (std::string e)
	{
		std::cout << "Gazebo error: Unable to connect to the position interface\n"
			<< e << "\n";
		return -1;
	}

	// Enable the motor
	std::cout << "Enable the motor\n" ;

	posIface->Lock(1);
	posIface->data->cmdEnableMotors = 1;
	posIface->Unlock();

	int i=0;
	while (i<10)
	{
		
		posIface->Lock(1);
// posIface->data->cmdVelocity.yaw = -0.1;
		if(i%2) {
			double vel=0.2*(i+1);
			posIface->data->cmdVelocity.pos.x = vel;
			std::cout << "Velocidad x="<< vel<<"\n";
		} else {
			std::cout << "Velocidad x=0\n";
			posIface->data->cmdVelocity.pos.x = 0;
		}
		posIface->Unlock();
		
		sleep(2);
		//		 usleep(100000);
		i++;
	}
	return 0;
}
