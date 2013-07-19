package sibtra.odometria;

import sibtra.imu.IMUEvent;

public interface OdometriaEventListener {
	
	public void handleOdometriaEvent(OdometriaEvent ev);

}
