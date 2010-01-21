/**
 * 
 */
package openservo;

import java.io.IOException;

/**
 * @author Alberto Hamilton
 *
 */
public class OpenServo {
	
	//Direcciones de los registros del Openservo
	
	/** Device type - 1 = OpenServo device type (Read Only ) */
	public static byte DEVICE_TYPE =0x00;
	/** Device subtype - 1 = OpenServo device subtype (Read Only ) */
	public static byte DEVICE_SUBTYPE =0x01;
	/** Major version number of OpenServo software (Read Only ) */
	public static byte VERSION_MAJOR =0x02;
	/** Minor version number of OpenServo software (Read Only ) */
	public static byte VERSION_MINOR =0x03;
	/** Flags high byte (Read Only ) */
	public static byte FLAGS_HI =0x04;
	/** Flags low byte (Read Only ) */
	public static byte FLAGS_LO =0x05;
	/** Timer high byte - incremented each ADC sample (Read Only ) */
	public static byte TIMER_HI =0x06;
	/** Timer low byte - incremented each ADC sample (Read Only ) */
	public static byte TIMER_LO =0x07;
	/** Servo position high byte (Read Only ) */
	public static byte POSITION_HI =0x08;
	/** Servo position low byte (Read Only ) */
	public static byte POSITION_LO =0x09;
	/** Servo velocity high byte (Read Only ) */
	public static byte VELOCITY_HI =0x0A;
	/** Servo velocity low byte (Read Only ) */
	public static byte VELOCITY_LO =0x0B;
	/** Servo power high byte (Read Only ) */
	public static byte POWER_HI =0x0C;
	/** Servo power low byte (Read Only ) */
	public static byte POWER_LO =0x0D;
	/** PWM clockwise value (Read Only ) */
	public static byte PWM_CW =0x0E;
	/** PWM counter-clockwise value (Read Only ) */
	public static byte PWM_CCW =0x0F;
	/** Seek position high byte (Read/Write ) */
	public static byte SEEK_HI =0x10;
	/** Seek position low byte (Read/Write ) */
	public static byte SEEK_LO =0x11;
	/** Speed seek position high byte (Read/Write ) */
	public static byte SEEK_VELOCITY_HI =0x12;
	/** Speed seek position low byte (Read/Write ) */
	public static byte SEEK_VELOCITY_LO =0x13;
	/** Battery Voltage value high byte (Read/Write ) */
	public static byte VOLTAGE_HI =0x14;
	/** Battery Voltage value low byte (Read/Write ) */
	public static byte VOLTAGE_LO =0x15;
	/** reserved curve data (Read/Write ) */
	public static byte CURVE_RESERVED =0x16;
	/** Remaining curve buffer space (Read/Write ) */
	public static byte CURVE_BUFFER =0x17;
	/** Curve Time delta high byte (Read/Write ) */
	public static byte CURVE_DELTA_HI =0x18;
	/** Curve Time delta low byte (Read/Write ) */
	public static byte CURVE_DELTA_LO =0x19;
	/** Curve position high byte (Read/Write ) */
	public static byte CURVE_POSITION_HI =0x1A;
	/** Curve position low byte (Read/Write ) */
	public static byte CURVE_POSITION_LO =0x1B;
	/** Curve in velocity high byte (Read/Write ) */
	public static byte CURVE_IN_VELOCITY_HI =0x1C;
	/** Curve in velocity low byte (Read/Write ) */
	public static byte CURVE_IN_VELOCITY_LO =0x1D;
	/** Curve out velocity high byte (Read/Write ) */
	public static byte CURVE_OUT_VELOCITY_HI =0x1E;
	/** Curve out velocity low byte (Read/Write ) */
	public static byte CURVE_OUT_VELOCITY_LO =0x1F;
	/** TWI address of servo (Read/Write Protected ) */
	public static byte TWI_ADDRESS =0x20;
	/** Programmable PID deadband value (Read/Write Protected ) */
	public static byte PID_DEADBAND =0x21;
	/** PID proportional gain high byte (Read/Write Protected ) */
	public static byte PID_PGAIN_HI =0x22;
	/** PID proportional gain low byte (Read/Write Protected ) */
	public static byte PID_PGAIN_LO =0x23;
	/** PID derivative gain high byte (Read/Write Protected ) */
	public static byte PID_DGAIN_HI =0x24;
	/** PID derivative gain low byte (Read/Write Protected ) */
	public static byte PID_DGAIN_LO =0x25;
	/** PID integral gain high byte (Read/Write Protected ) */
	public static byte PID_IGAIN_HI =0x26;
	/** PID integral gain low byte (Read/Write Protected ) */
	public static byte PID_IGAIN_LO =0x27;
	/** PWM frequency divider high byte (Read/Write Protected ) */
	public static byte PWM_FREQ_DIVIDER_HI =0x28;
	/** PWM frequency divider low byte (Read/Write Protected ) */
	public static byte PWM_FREQ_DIVIDER_LO =0x29;
	/** Minimum seek position high byte (Read/Write Protected ) */
	public static byte MIN_SEEK_HI =0x2A;
	/** Minimum seek position low byte (Read/Write Protected ) */
	public static byte MIN_SEEK_LO =0x2B;
	/** Maximum seek position high byte (Read/Write Protected ) */
	public static byte MAX_SEEK_HI =0x2C;
	/** Maximum seek position low byte (Read/Write Protected ) */
	public static byte MAX_SEEK_LO =0x2D;
	/** Reverse seek sense (Read/Write Protected ) */
	public static byte REVERSE_SEEK =0x2E;
	/** (Read/Write Protected ) */
	public static byte RESERVED =0x2F;

	//Comandos del OpenServo
	/** Reset microcontroller  */
	public static byte RESET=(byte)0x80;
	/** Read/Write registers with simple checksum  */
	public static byte CHECKED_TXN=(byte)0x81;
	/** Enable PWM to motors  */
	public static byte PWM_ENABLE=(byte)0x82;
	/** Disable PWM to servo motors  */
	public static byte PWM_DISABLE=(byte)0x83;
	/** Enable write of read/write protected registers  */
	public static byte WRITE_ENABLE=(byte)0x84;
	/** Disable write of read/write protected registers  */
	public static byte WRITE_DISABLE=(byte)0x85;
	/** Save read/write protected registers fo EEPROM  */
	public static byte REGISTERS_SAVE=(byte)0x86;
	/** Restore read/write protected registers from EEPROM  */
	public static byte REGISTERS_RESTORE=(byte)0x87;
	/** Restore read/write protected registers to defaults  */
	public static byte REGISTERS_DEFAULT=(byte)0x88;
	/** Erase the AVR EEPROM  */
	public static byte EEPROM_ERASE=(byte)0x89;
	/** Request a new Voltage sample  */
	public static byte VOLTAGE_READ=(byte)0x90;
	/** Enable curve based motion  */
	public static byte CURVE_MOTION_ENABLE=(byte)0x91;
	/** Disable curve based motion  */
	public static byte CURVE_MOTION_DISABLE=(byte)0x92;
	/** Clear the curve buffer  */
	public static byte CURVE_MOTION_RESET=(byte)0x93;
	/** Append a new curve  */
	public static byte CURVE_MOTION_APPEND=(byte)0x94;

	
	//Campos de cada instancia
	/** En que adaptador OSIF está conectado */
	int adaptador;
	
	//Campos de configuracion, Se leeran al iniciailizar
	/** Que identificador tiene TWI_ADDRESS */
	int id;
	
	/** Descripcion para saber que servo es */
	String Descripcion="Sin nombre";
	
	/** Posicion minima programada MIN_SEEK  @see <a href='http://www.openservo.org/APIServoConfigure'>configuracion</a>*/ 
	int minPosicion;
	
	/** Posición máxima programada MAX_SEEK  @see <a href='http://www.openservo.org/APIServoConfigure'>configuracion</a>*/
	int maxPosicion;
	
	/** PID_DEADBAND @see <a href='http://www.openservo.org/APIServoConfigure'>configuracion</a>*/
	int bandaMuerta;
	
	/** PID_PGAIN */
	int Kp;
	
	/** PID_IGAIN */
	int Ki;
	
	/** PID_DGAIN */
	int Kd;
	
	/** PWM_FREQ_DIVIDER */
	int divisorPWM;
	
	/** REVERSE_SEEK @see <a href='http://www.openservo.org/APIServoConfigure'>configuracion</a>*/
	boolean movimientoInvertido;
	
	/** Si las variables tienen actualizada la configuración, true cuando se completa {@link #leeConfiguracion()} */
	private boolean configuracionLeida=false;

	//el resto de los valores se leen y ecriben en tiempo de ejecución

	/** Se trata de contactar con el OpenServo y se le piden todas las variables de configuración */
	public OpenServo(int adap, int idI2C) throws IOException {
		adaptador=adap;
		id=idI2C;
		//vemos si dispositivo vivo
		if(!OsifJNI.OSIF_probe(adaptador, id )) {
			System.err.println("\nEl dispositivo "+id+" NO respondió");
			throw new IllegalArgumentException("\nEl dispositivo "+id+" NO respondió");
		}	
		//Vemos si es OpenServo
		byte[] buffer=new byte[4];
		if(OsifJNI.OSIF_read(adaptador, id, (byte) 0x00, buffer, 1)<0) {
			System.err.println("\n Error al leer de dispositivo "+id);
			throw new IllegalArgumentException("\nEl dispositivo "+id+" NO se puede leer");
		}
		if(buffer[0]!=0x01) {
			System.err.println("\nEl dispositivo "+id+" NO es OpenServo");
			throw new IllegalArgumentException("\nEl dispositivo "+id+" NO es OpenServo");
		}

		leeConfiguracion();
		
	}
	
	private int buff2int(byte[] buff,int pos) {
		return ((((int)buff[pos])&0xff)<<8) + (((int)buff[pos+1])&0xff);
	}
	
	private int buff2int(byte[] buff) {
		return buff2int(buff, 0);
	}

	private void int2buff(int val, byte[] buff, int pos) {
		buff[pos]=(byte)((val&0xff00)>>8);
		buff[pos]=(byte)(val&0xff);
	}

	private void int2buff(int val, byte[] buff) {
		int2buff(val, buff, 0);
	}
	
	private byte[] int2buff(int val) {
		byte[] buff={(byte)((val&0xff00)>>8), (byte)((val&0xff00)>>8)};
		return buff;
	}

	
	/** Actuliza todos los campos correspondientes a las variables de configuración */
	public void leeConfiguracion() throws IOException {
		byte dini=PID_DEADBAND;
		byte dfin=REVERSE_SEEK;
		int len=dfin-dini+1;
		byte buffer[]=new byte[len];
		if(OsifJNI.OSIF_read(adaptador, id, dini, buffer, len)<0) {
			System.err.println("\n Error al leer configuración del dispositivo "+id);
			configuracionLeida=false;
			throw new IOException("Error al leer configuración del dispositivo "+id);
		}
		//pasamos a obtener todos los campos
		minPosicion=buff2int(buffer, MIN_SEEK_HI-dini);
		maxPosicion=buff2int(buffer, MAX_SEEK_HI-dini);
		
		Kp=buff2int(buffer, PID_PGAIN_HI-dini);
		Ki=buff2int(buffer, PID_IGAIN_HI-dini);
		Kd=buff2int(buffer, PID_DGAIN_HI-dini);
		
		divisorPWM=buff2int(buffer, PWM_FREQ_DIVIDER_HI-dini);

		movimientoInvertido=(buffer[REVERSE_SEEK-dini]!=0);
		
		bandaMuerta=buffer[PID_DEADBAND-dini];
		configuracionLeida=true;
	}
	
	//Envio de comandos
	
	/** Resetear OS */
	public void reset() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, RESET)<0) {
			throw new IOException("Error al tratar de hacer RESET a dispositivo "+id);
		}
	}

	/** Salva configuración en EEPROM del OS */
	public void salvaConfiguracion() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, REGISTERS_SAVE)<0) {
			throw new IOException("Error al tratar de hacer salvar configuración a dispositivo "+id);
		}		
	}

	/** Habilitar PWM */
	public void habilitaPWM() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, PWM_ENABLE)<0) {
			throw new IOException("Error al tratar de habilitar PWM en dispositivo "+id);
		}		
	}

	/** Deshabilitar PWM */
	public void deshabilitarPWM() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, PWM_DISABLE)<0) {
			throw new IOException("Error al tratar de deshabilitar PWM en dispositivo "+id);
		}		
	}

	/** Permitir escritura de registros protegidos */
	public void permitirEscrituraProtegidos() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, WRITE_ENABLE)<0) {
			throw new IOException("Error al tratar de permitir la escritura en registros protegidos dispositivo "+id);
		}		
	}

	/** prohibir escritura de registros protegidos */
	public void prohibirEscrituraProtegidos() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, WRITE_DISABLE)<0) {
			throw new IOException("Error al tratar prohibir la escritura en registros protegidos dispositivo "+id);
		}		
	}

	/** Restaura configuración desde EEPROM del OS */
	public void restaurarConfiguracion() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, REGISTERS_RESTORE)<0) {
			throw new IOException("Error al tratar de restaurar configuración desde EEPROM a dispositivo "+id);
		}		
	}

	/** Salva configuración en EEPROM del OS */
	public void restauraConfiguracionPorDefecto() throws IOException {
		if(OsifJNI.OSIF_command(adaptador, id, REGISTERS_DEFAULT)<0) {
			throw new IOException("Error al tratar de restaurar configuración por defecto en dispositivo "+id);
		}		
	}


	// valores de configuración
	/**
	 * @return the adaptador
	 */
	public int getAdaptador() throws IOException{
		return adaptador;
	}

	/**
	 * @return the bandaMuerta
	 */
	public int getBandaMuerta() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return bandaMuerta;
	}

	/**
	 * @param bandaMuerta the bandaMuerta to set
	 */
	public void setBandaMuerta(int bandaMuerta) throws IOException {
		permitirEscrituraProtegidos();
		byte buffer[]= {(byte)(bandaMuerta&0xff)};
		if(OsifJNI.OSIF_write(adaptador, id, PID_DEADBAND, buffer, 1)<0) {
			throw new IOException("Error al tratar de fijar la banda muerta en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		this.bandaMuerta = bandaMuerta;
	}

	/**
	 * @return the descripcion
	 */
	public String getDescripcion() throws IOException {
		return Descripcion;
	}

	/**
	 * @param descripcion the descripcion to set
	 */
	public void setDescripcion(String descripcion) throws IOException {
		Descripcion = descripcion;
	}

	/**
	 * @return the divisorPWM
	 */
	public int getDivisorPWM() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return divisorPWM;
	}

	/**
	 * @param divisorPWM the divisorPWM to set
	 */
	public void setDivisorPWM(int divisorPWM) throws IOException {
		permitirEscrituraProtegidos();
		if(OsifJNI.OSIF_write(adaptador, id, PWM_FREQ_DIVIDER_HI, int2buff(divisorPWM), 2)<0) {
			throw new IOException("Error al tratar de fijar divisor PWM en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		this.divisorPWM = divisorPWM;
	}

	/**
	 * @return the id
	 */
	public int getId() throws IOException {
		return id;
	}

	/**
	 * @return the kd
	 */
	public int getKd() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return Kd;
	}

	/**
	 * @param kd the kd to set
	 */
	public void setKd(int kd) throws IOException {
		permitirEscrituraProtegidos();
		if(OsifJNI.OSIF_write(adaptador, id, PID_DGAIN_HI, int2buff(kd), 2)<0) {
			throw new IOException("Error al tratar de fijar Kd en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		Kd = kd;
	}

	/**
	 * @return the ki
	 */
	public int getKi() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return Ki;
	}

	/**
	 * @param ki the ki to set
	 */
	public void setKi(int ki) throws IOException {
		permitirEscrituraProtegidos();
		if(OsifJNI.OSIF_write(adaptador, id, PID_IGAIN_HI, int2buff(ki), 2)<0) {
			throw new IOException("Error al tratar de fijar Ki en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		Ki = ki;
	}

	/**
	 * @return the kp
	 */
	public int getKp() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return Kp;
	}

	/**
	 * @param kp the kp to set
	 */
	public void setKp(int kp) throws IOException {
		permitirEscrituraProtegidos();
		if(OsifJNI.OSIF_write(adaptador, id, PID_PGAIN_HI, int2buff(kp), 2)<0) {
			throw new IOException("Error al tratar de fijar Kp en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		Kp = kp;
	}

	/**
	 * @return the maxPosicion
	 */
	public int getMaxPosicion() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return maxPosicion;
	}

	/**
	 * @param maxPosicion the maxPosicion to set
	 */
	public void setMaxPosicion(int maxPosicion) throws IOException {
		permitirEscrituraProtegidos();
		if(OsifJNI.OSIF_write(adaptador, id, MAX_SEEK_HI, int2buff(maxPosicion), 2)<0) {
			throw new IOException("Error al tratar de fijar maxima posición en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		this.maxPosicion = maxPosicion;
	}

	/**
	 * @return the minPosicion
	 */
	public int getMinPosicion() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return minPosicion;
	}

	/**
	 * @param minPosicion the minPosicion to set
	 */
	public void setMinPosicion(int minPosicion) throws IOException {
		permitirEscrituraProtegidos();
		if(OsifJNI.OSIF_write(adaptador, id, MIN_SEEK_HI, int2buff(minPosicion), 2)<0) {
			throw new IOException("Error al tratar de fijar posicón mínima en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		this.minPosicion = minPosicion;
	}

	/**
	 * @return the movimientoInvertido
	 */
	public boolean isMovimientoInvertido() throws IOException {
		if(!configuracionLeida)
			leeConfiguracion();
		return movimientoInvertido;
	}

	/**
	 * @param movimientoInvertido the movimientoInvertido to set
	 */
	public void setMovimientoInvertido(boolean movimientoInvertido) throws IOException {
		permitirEscrituraProtegidos();
		byte[] buff={(movimientoInvertido?(byte)1:(byte)0)};
		if(OsifJNI.OSIF_write(adaptador, id, REVERSE_SEEK, buff, 1)<0) {
			throw new IOException("Error al tratar de fijar movimiento invertido en dispositivo "+id);
		}		
		prohibirEscrituraProtegidos();
		this.movimientoInvertido = movimientoInvertido;
	}
	
	//parámetros vivos de sólo lectura
	
	public int getTimer() throws IOException {
		byte[] buff=new byte[2];
		if(OsifJNI.OSIF_read(adaptador, id, TIMER_HI, buff, 2)<0)
			throw new IOException("Error al leer Timer del dispositivo "+id);
		return buff2int(buff);
	}
	
	public int getVelocity() throws IOException {
		byte[] buff=new byte[2];
		if(OsifJNI.OSIF_read(adaptador, id, VELOCITY_HI, buff, 2)<0)
			throw new IOException("Error al leer velocidad del dispositivo "+id);
		return buff2int(buff);
	}
	
	public int getPosition() throws IOException {
		byte[] buff=new byte[2];
		if(OsifJNI.OSIF_read(adaptador, id, POSITION_HI, buff, 2)<0)
			throw new IOException("Error al leer Posicion del dispositivo "+id);
		return buff2int(buff);
	}
	
	public int getCorriente() throws IOException {
		byte[] buff=new byte[2];
		if(OsifJNI.OSIF_read(adaptador, id, POWER_HI, buff, 2)<0)
			throw new IOException("Error al leer corriente del dispositivo "+id);
		return buff2int(buff);
	}
	
	public int getPWMDerecha() throws IOException {
		byte[] buff=new byte[1];
		if(OsifJNI.OSIF_read(adaptador, id, PWM_CW, buff, 1)<0)
			throw new IOException("Error al leer PWM derecha del dispositivo "+id);
		return (int)buff[0];
	}
	
	public int getPWMIzquierda() throws IOException {
		byte[] buff=new byte[1];
		if(OsifJNI.OSIF_read(adaptador, id, PWM_CCW, buff, 1)<0)
			throw new IOException("Error al leer PWM izda. del dispositivo "+id);
		return (int)buff[0];
	}

	/** Votaje @see <a href='http://www.openservo.org/APIServoGetVoltage'>get voltaje</a>*/
	public int getVoltaje() throws IOException {
		//primero tenemos que mandar comando de petición de voltaje
		if(OsifJNI.OSIF_command(adaptador, id, VOLTAGE_READ)<0) 
			throw new IOException("Error al tratar de solicitar voltaje en dispositivo "+id);
		byte[] buff=new byte[2];
		if(OsifJNI.OSIF_read(adaptador, id, POWER_HI, buff, 2)<0)
			throw new IOException("Error al el voltaje del dispositivo "+id);
		return buff2int(buff);
	}
	


	// Parametros vivos de lectura y escritura
	public int getPosicionDeseada() throws IOException {
		byte[] buff=new byte[2];
		if(OsifJNI.OSIF_read(adaptador, id, SEEK_HI, buff, 2)<0)
			throw new IOException("Error al leer Posicion Deseada del dispositivo "+id);
		return buff2int(buff);
	}
	
	public void setPosicionDeseada(int pos) throws IOException {
		if(OsifJNI.OSIF_write(adaptador, id, SEEK_HI, int2buff(pos), 2)<0)
			throw new IOException("Error al fijar Posicion Deseada del dispositivo "+id);
	}
	
	
	



}
